"""Held-out paired first-recommendation response diagnostics.

Differences describe a forced first recommendation followed by the declared
policy continuation. Acceptance is a scored outcome, never a model input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from shield.predictor import PointForecast
from .branches import load_branch_groups, episode_family
from .data import claim_directory
from .model import atomic_json, load_predictor, sha256_file
from .validate import artifact_hashes, audit_holdout, request_at, validation_source_hash


def anchor_mean(values, mask):
    """Mean horizons within candidates, candidates within anchors, then anchors."""
    counts = mask.sum(axis=-1)
    pair_means = np.divide(np.where(mask, values, 0).sum(axis=-1), counts,
                           out=np.zeros_like(counts, dtype=float), where=counts > 0)
    pairs_per_anchor = (counts > 0).sum(axis=-1)
    means = np.divide(pair_means.sum(axis=-1), pairs_per_anchor,
                     out=np.zeros_like(pairs_per_anchor, dtype=float), where=pairs_per_anchor > 0)
    return float(means[pairs_per_anchor > 0].mean()) if np.any(pairs_per_anchor > 0) else None


def summarize_responses(predictions, targets, valid, actions, accepted):
    """Score complete candidate grids, retaining partial-horizon coverage.

    predictions/targets/valid: [anchors,25,horizon]; actions: [25,2];
    accepted: [anchors,25,2], first-step authoritative acceptance flags.
    Missing targets may use arbitrary padding; only valid values are scored.
    """
    predictions, targets = np.asarray(predictions, dtype=float), np.asarray(targets, dtype=float)
    valid, actions, accepted = np.asarray(valid), np.asarray(actions), np.asarray(accepted)
    if predictions.ndim != 3 or predictions.shape != targets.shape or predictions.shape != valid.shape:
        raise ValueError("response predictions, targets and mask require matching [anchors,25,horizon] shapes")
    if not predictions.shape[0] or not predictions.shape[2] or valid.dtype != bool or accepted.dtype != bool:
        raise ValueError("response data must have anchors/horizons and boolean observation/acceptance masks")
    if (actions.shape != (25, 2) or actions.dtype.kind not in "iu" or
            {tuple(row) for row in actions.tolist()} != {(b, m) for b in range(5) for m in range(5)}):
        raise ValueError("response scoring requires the complete unique 25-action grid")
    if predictions.shape[1] != 25 or accepted.shape != predictions.shape[:2] + (2,):
        raise ValueError("response acceptance must align with anchors and candidates")
    if not np.isfinite(predictions).all() or not np.isfinite(targets[valid]).all():
        raise ValueError("response predictions and observed targets must be finite")
    if np.any(valid[..., 1:] & ~valid[..., :-1]):
        raise ValueError("observed branch horizons must be contiguous prefixes")
    if np.any(accepted & (actions[None] == 0)):
        raise ValueError("zero recommendations cannot have accepted components")
    control = int(np.flatnonzero(np.all(actions == 0, axis=-1))[0])
    response = predictions - predictions[:, control:control + 1]
    observed_response = np.where(valid, targets, 0) - np.where(valid, targets, 0)[:, control:control + 1]
    paired_mask = valid & valid[:, control:control + 1]
    absolute_error = np.abs(predictions - np.where(valid, targets, 0))
    response_error = np.abs(response - observed_response)
    magnitude = np.abs(observed_response)
    active = actions > 0
    nonzero = np.broadcast_to(active.any(axis=-1), predictions.shape[:2])
    any_accepted = accepted.any(axis=-1)
    strata = {
        "all_nonzero": nonzero,
        "accepted_pure_bolus": (active[:, 0] & ~active[:, 1])[None] & accepted[..., 0],
        "accepted_pure_meal": (~active[:, 0] & active[:, 1])[None] & accepted[..., 1],
        "joint_recommendation": np.broadcast_to(active.all(axis=-1), predictions.shape[:2]),
        "any_active_component_rejected": ((active[None] & ~accepted).any(axis=-1)),
        "no_first_component_accepted": nonzero & ~any_accepted,
        "noop": np.broadcast_to(~active.any(axis=-1), predictions.shape[:2]),
    }
    summaries = {}
    for name, select in strata.items():
        mask = paired_mask & select[..., None]
        pair_observed = mask.any(axis=-1)
        row = {"anchors": int(pair_observed.any(axis=-1).sum()),
               "candidate_pairs": int(pair_observed.sum()), "observed_horizon_points": int(mask.sum())}
        if mask.any():
            error, baseline = anchor_mean(response_error, mask), anchor_mean(magnitude, mask)
            row.update(response_mae=error, zero_response_mae=baseline,
                       absolute_mae=anchor_mean(absolute_error, valid & select[..., None]))
            if baseline > 0:
                row["improvement_over_zero_pct"] = 100 * (baseline - error) / baseline
            informative = mask & (magnitude > 5)
            row["sign_points_over_5_mg_dl"] = int(informative.sum())
            if informative.any():
                row["sign_agreement_over_5_mg_dl"] = anchor_mean(
                    np.sign(response) == np.sign(observed_response), informative)
        summaries[name] = row
    curves = []
    for index in sorted(range(25), key=lambda value: tuple(actions[value])):
        mask = paired_mask[:, index]
        counts = mask.sum(axis=0)
        def curve(values):
            return [float(values[mask[:, h], h].mean()) if counts[h] else None
                    for h in range(predictions.shape[-1])]
        curves.append({"action": actions[index].tolist(), "anchors_by_horizon": counts.tolist(),
                       "predicted_response_by_horizon": curve(response[:, index]),
                       "realized_response_by_horizon": curve(observed_response[:, index]),
                       "response_mae_by_horizon": curve(response_error[:, index])})
    return {"strata": summaries, "dose_level_curves": curves, "absolute_mae": anchor_mean(absolute_error, valid),
            "observed_horizon_points": int(valid.sum()), "requested_horizon_points": int(valid.size),
            "observed_coverage_fraction": float(valid.mean()),
            "weighting": "equal horizons within each candidate, equal candidates within each anchor, equal anchors",
            "scope": "paired forced-first-recommendation differences under fixed continuation; overlapping anchors are not independent replicates"}


def collection_inputs(paths, horizon):
    """Require complete collections so unreachable anchors stay in coverage."""
    roots = [Path(path).resolve() for path in paths]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("response evaluation requires unique complete collection directories")
    groups = load_branch_groups(roots, required_split="test")
    rows, hashes, families = [], {}, set()
    for root in roots:
        record = json.loads((root / "collection.json").read_text())
        if record["split"] != "test" or record["metadata"]["split"] != "test":
            raise ValueError("response evaluation requires test collections")
        family = episode_family(record)
        if family in families:
            raise ValueError("response collections repeat a reset family")
        families.add(family)
        subset = [group for group in groups if group.path.parent == root]
        planned = record["anchors_requested"]
        observed = [group.metadata["anchor"] for group in subset]
        skipped = [item["anchor"] for item in record["skipped_anchors"]]
        if (not planned or any(type(anchor) is not int or anchor < 1 for anchor in planned)
                or len(set(planned)) != len(planned) or len(set(observed + skipped)) != len(observed + skipped)
                or set(planned) != set(observed + skipped)):
            raise ValueError("observed and skipped anchors must partition the complete requested set")
        if any(group.metadata["horizon_steps"] != horizon for group in subset):
            raise ValueError("branch/model horizon mismatch")
        if sha256_file(root / "prefix.npz") != record["prefix_sha256"]:
            raise ValueError("collection prefix hash mismatch")
        for name in ("collection.json", "run.json", "prefix.npz", "metrics.json", "split_check.json"):
            hashes[str(root / name)] = sha256_file(root / name)
        for group in subset:
            for name in ("data.npz", "metadata.json", "audit.json", "manifest.json"):
                hashes[str(group.path / name)] = sha256_file(group.path / name)
        rows.append({"path": str(root), "patient_type": family[0], "patient_name": family[1], "seed": family[2],
                     "anchors_requested": planned, "anchors_observed": observed, "skipped_anchors": record["skipped_anchors"],
                     "requested_horizon_points": len(planned) * 25 * horizon,
                     "metadata": record["metadata"], "prefix_sha256": record["prefix_sha256"]})
    if not groups:
        raise ValueError("no supplied collection has an observed branch anchor")
    return groups, rows, hashes


def audit_branch_holdout(artifact, groups, collections):
    # Reuse the same reset-family and continuation gate as factual evaluation.
    # Prefix groups are causal inputs, not factual continuation episodes.
    items = []
    for group in groups:
        meta = dict(group.metadata, episode_id=group.metadata["group_id"],
                    data_sha256=group.metadata["prefix_data_sha256"])
        items.append(SimpleNamespace(metadata=meta))
    for row in collections:  # Includes reset families with no reachable anchor.
        meta = dict(row["metadata"], data_sha256=row["prefix_sha256"])
        items.append(SimpleNamespace(metadata=meta))
    result = audit_holdout(artifact, items)
    return {**result, "checked_branch_groups": len(groups), "checked_collections": len(collections)}


def _forecast(adapter, request, horizon):
    result = adapter.forecast(request)
    if not isinstance(result, PointForecast):
        raise ValueError("branch adapter unavailable after declared warm-up")
    values = result.glucose_mg_dl.detach().cpu().numpy().copy()
    if values.shape != (25, horizon) or not np.isfinite(values).all():
        raise ValueError("branch adapter returned invalid absolute point forecasts")
    return values


def _evaluate_responses(artifact_dir, collection_dirs, output, permutation_seed):
    before, source_before = artifact_hashes(artifact_dir), validation_source_hash()
    adapter = load_predictor(artifact_dir, device="cpu")
    horizon, required = adapter.config.horizon_steps, adapter.metadata.history_length
    groups, collections, inputs_before = collection_inputs(collection_dirs, horizon)
    split_check = audit_branch_holdout(adapter.artifact, groups, collections)
    actions = np.asarray(groups[0].candidates)
    count = len(groups)
    predictions = np.empty((count, 25, horizon))
    ablated = np.empty_like(predictions)
    targets = np.full_like(predictions, np.nan)
    valid = np.zeros(predictions.shape, dtype=bool)
    accepted = np.zeros((count, 25, 2), dtype=bool)
    latency, order_errors, rows = [], [], []
    for index, group in enumerate(groups):
        if not np.array_equal(actions, group.candidates):
            raise ValueError("candidate grid order differs between groups")
        prefix = SimpleNamespace(**group.prefix, metadata=group.metadata)
        anchor = group.metadata["anchor"]
        candidates = tuple(map(tuple, actions.tolist()))
        request = request_at(prefix, anchor, required, candidates)
        adapter.reset()
        start = time.perf_counter()
        predictions[index] = _forecast(adapter, request, horizon)
        latency.append(time.perf_counter() - start)
        adapter.reset()
        reverse = _forecast(adapter, request_at(prefix, anchor, required, candidates[::-1]), horizon)[::-1]
        order_errors.append(float(np.max(np.abs(reverse - predictions[index]))))
        if not np.allclose(reverse, predictions[index], atol=1e-4, rtol=1e-6):
            raise ValueError("candidate-order permutation changed the forecast")
        adapter.reset()
        ablated[index] = _forecast(adapter, request_at(prefix, anchor, required, ((0, 0),) * 25), horizon)
        if not np.allclose(ablated[index], ablated[index, :1], atol=1e-4, rtol=1e-6):
            raise ValueError("identical candidate inputs produced different forecasts")
        for candidate, branch in enumerate(group.branches):
            length = len(branch["rewards"])
            targets[index, candidate, :length] = branch["observations"][1:, 0]
            valid[index, candidate, :length] = True
            accepted[index, candidate] = branch["accepted"][0]
        rows.append({"group_id": group.metadata["group_id"], "path": str(group.path),
                     "patient_type": group.metadata["patient_type"], "patient_name": group.metadata["patient_name"],
                     "seed": group.metadata["seed"], "anchor": anchor,
                     "current_cgm": float(prefix.observations[-1, 0]),
                     "data_sha256": group.metadata["data_sha256"]})
    summary = summarize_responses(predictions, targets, valid, actions, accepted)
    requested = sum(row["requested_horizon_points"] for row in collections)
    summary.update(requested_horizon_points=requested, observed_coverage_fraction=int(valid.sum()) / requested,
                   anchors_requested=sum(len(row["anchors_requested"]) for row in collections), anchors_observed=count)
    control = int(np.flatnonzero(np.all(actions == 0, axis=-1))[0])
    nonzero = np.flatnonzero(np.any(actions != 0, axis=-1))
    rng = np.random.Generator(np.random.PCG64(permutation_seed))
    permutations = np.tile(np.arange(25), (count, 1))
    for row in permutations:
        row[nonzero] = rng.permutation(nonzero)
    shuffled_targets = np.take_along_axis(targets, permutations[..., None], axis=1)
    shuffled_valid = np.take_along_axis(valid, permutations[..., None], axis=1)
    shuffled_error = np.abs((predictions - predictions[:, control:control + 1]) -
                           (shuffled_targets - shuffled_targets[:, control:control + 1]))
    shuffled_mask = shuffled_valid & shuffled_valid[:, control:control + 1]
    shuffled_mask[:, control] = False
    diagnostics = {"candidate_order_max_absolute_error": max(order_errors),
                   "candidate_order_atol": 1e-4, "candidate_order_rtol": 1e-6,
                   "candidate_ablation": summarize_responses(ablated, targets, valid, actions, accepted),
                   "label_permutation_seed": permutation_seed,
                   "label_permutation_response_mae": anchor_mean(shuffled_error, shuffled_mask),
                   "interpretation": "Descriptive controls; no threshold for a shuffle-score change was registered."}
    diagnostics["candidate_ablation"].update(requested_horizon_points=requested,
        observed_coverage_fraction=int(valid.sum()) / requested)
    by_patient = {}
    for patient in sorted({(row["patient_type"], row["patient_name"]) for row in collections}):
        mask = np.array([(row["patient_type"], row["patient_name"]) == patient for row in rows])
        result = (summarize_responses(predictions[mask], targets[mask], valid[mask], actions, accepted[mask])
                  if mask.any() else {"anchors_observed": 0, "observed_horizon_points": 0,
                                      "reason": "no reachable branch anchors; error metrics unavailable"})
        denominator = sum(row["requested_horizon_points"] for row in collections
                          if (row["patient_type"], row["patient_name"]) == patient)
        result.update(requested_horizon_points=denominator, observed_coverage_fraction=int(valid[mask].sum()) / denominator)
        by_patient["/".join(patient)] = result
    with (output / "responses.npz").open("xb") as stream:
        np.savez_compressed(stream, predictions=predictions, targets=targets, valid=valid, actions=actions,
                            first_accepted=accepted, candidate_ablated=ablated, label_permutations=permutations,
                            latency_seconds=latency, candidate_order_errors=order_errors)
    if (before != artifact_hashes(artifact_dir) or source_before != validation_source_hash()
            or any(sha256_file(path) != digest for path, digest in inputs_before.items())):
        raise ValueError("artifact, source or branch inputs changed during response evaluation")
    report = {"schema": "glucoalg.response_validation/1", "artifact_dir": str(Path(artifact_dir).resolve()),
              "artifact_hashes": before, "evaluation_source_sha256": source_before, "input_hashes": inputs_before,
              "horizon_steps": horizon, "controller_interval_minutes": 5,
              "continuation_policy": adapter.metadata.continuation_policy,
              "groups": rows, "collections": [{key: value for key, value in row.items() if key != "metadata"} for row in collections],
              "summary": summary, "by_patient": by_patient, "diagnostics": diagnostics,
              "latency_seconds": {"p50": float(np.quantile(latency, .5)), "p99": float(np.quantile(latency, .99)), "max": max(latency)},
              "responses_sha256": sha256_file(output / "responses.npz")}
    atomic_json(output / "report.json", report)
    atomic_json(output / "split_check.json", split_check)
    atomic_json(output / "metrics.json", {"groups": count, "requested_horizon_points": requested,
                "observed_horizon_points": int(valid.sum()), "coverage_fraction": int(valid.sum()) / requested,
                "absolute_mae": summary["absolute_mae"],
                "response_mae": summary["strata"]["all_nonzero"]["response_mae"],
                "zero_response_mae": summary["strata"]["all_nonzero"]["zero_response_mae"],
                "latency_p99_seconds": report["latency_seconds"]["p99"]})
    return report


def evaluate_responses(artifact_dir, collection_dirs, output_dir, *, permutation_seed=4101):
    if type(permutation_seed) is not int or not 0 <= permutation_seed < 2**32:
        raise ValueError("permutation seed must be an unsigned32-bit integer")
    output = claim_directory(output_dir)
    atomic_json(output / "run.json", {"status": "started"})
    try:
        report = _evaluate_responses(artifact_dir, collection_dirs, output, permutation_seed)
        atomic_json(output / "run.json", {"status": "complete", "report_sha256": sha256_file(output / "report.json")})
        return report
    except BaseException as exc:
        atomic_json(output / "run.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--collections", nargs="+", required=True, help="complete test collection directories, including skipped anchors")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutation-seed", type=int, default=4101)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = evaluate_responses(args.artifact, args.collections, args.output_dir, permutation_seed=args.permutation_seed)
    print(json.dumps(report["summary"], allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
