# Evaluation results: 2026-10-08 snapshot

This folder contains the completed shield comparison for **CPO, PPO-Lag,
TRPOLag, RCPO, CRPO, PCPO, CUP and FOCOPS**. The underlying runs are frozen;
this release shares existing results without running new experiments.

## Coverage

| Dimension | Values |
| --- | --- |
| Diabetes configurations | T1D, T2D, T2D without pump |
| Age cohorts | Child, adolescent, adult |
| Policy training seeds | 100, 101, 102: **three seeds per algorithm/configuration** |
| Action modes | Deterministic and stochastic, reported separately |
| Final generalization patients | Unseen same-cohort patients #002–#010 |
| Final generalization horizon | Seven requested days |
| Controls | No shield and static shield |
| Predictive arms | Fixed absolute penalty 10; validation-selected absolute penalty |
| Penalty selection | 1, 5, 10, 15, 20 on seen patient #001, one-day validation only |

Across all eight algorithms, the fixed-10 unseen-seven-day comparison has
**144 complete panels**: 72 algorithm/type/cohort cells per action mode.
The selected-penalty comparison has **108 numeric panels and 36 explicit
`no_eligible_penalty` panels**. These NA entries are completed selection
outcomes, not pending evaluations. No-shield and static controls are complete.
CRPO is stored as `OnCRPO` in the implementation and machine-readable files.

Predictive panels use three policy seeds × three predictor seeds × nine unseen
patients × three evaluation resets: **243 physical episodes**. A no-shield or
static panel uses **81 physical episodes**. Predictor seeds are 3101–3103 and
evaluation reset/action seeds are 83000–83002. Reusing a matched control or a
selected-10 alias does not create additional physical episodes.

Reported means weight resets, patients, predictor seeds within policy, and
policy seeds equally. The reported SD is the **sample SD across three policy
means**, not across all episodes. Primary TIR uses the requested horizon as its
denominator; read it alongside coverage and early/physiological termination.

## Files

| File | Contents |
| --- | --- |
| [comparison-tables.md](tables/comparison-tables.md) | Human-readable comparisons, safety/coverage and generalization degradation |
| [matched-unseen7.csv](tables/matched-unseen7.csv) | 216 matched predictive rows for the six added algorithms; no/static controls included |
| [coverage-unseen7.csv](tables/coverage-unseen7.csv) | Support counts and scientific NA reasons for those 216 rows |
| [all-panels.csv](tables/all-panels.csv) | 1,728 six-algorithm panels: seen/unseen patients × one/seven days × shield arms × action modes |
| [historical-cpo-ppolag-unseen7.csv](tables/historical-cpo-ppolag-unseen7.csv) | 144 historical CPO/PPO-Lag rows, retaining their distinct source provenance |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Fields, units, aggregation and reuse notes |
| [MANIFEST.json](MANIFEST.json) | Exact file sizes and SHA256 checksums |

All five table files are byte-identical to the frozen comparison export. Their
combined size is approximately **7 MB**. The source-summary SHA is recorded in
`source-snapshot.json`; large internal provenance ledgers are not required to
read the tables.

Any original server path in the preserved table footer is archival provenance,
not a download location. Use the public links below.

## Download episode metrics

The matching [Hugging Face evaluation folder](https://huggingface.co/datasets/safe-diabetes-benchmark/transition_datasets/tree/evaluations-2026-10-08/evaluations/2026-10-08)
contains these tables plus **180,360 retained physical episode metric rows**,
split into eight compressed CSV files. It includes seen and unseen patients,
one- and seven-day horizons, both action modes, all policy/predictor/reset
seeds, and the penalties actually evaluated in the retained outcome campaign.
It is separate from that repository's predictor-training data.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "safe-diabetes-benchmark/transition_datasets",
    repo_type="dataset",
    revision="evaluations-2026-10-08",
    allow_patterns=["evaluations/2026-10-08/*"],
    local_dir="diabetes-results",
)
```

This downloads the evaluation release without the predictor-training shards or
model weights. The folder also provides a single ZIP download and checksums.
See its README for a version-pinned command and episode-level loading examples.

For the summary tables already in this checkout:

```python
import json
from pathlib import Path
import pandas as pd

root = Path("benchmark-results/2026-10-08/tables")
matched = pd.read_csv(root / "matched-unseen7.csv", keep_default_na=False)
numeric = matched[matched["status"] == "complete"].copy()
numeric["predictive_tir_pct"] = numeric["metrics"].map(
    lambda text: json.loads(text)["requested_tir_pct"]["mean"]
)
```

## Provenance and limits

CPO/PPO-Lag retain historical runtime results; the other original algorithms
and repaired PCPO have separate frozen source histories. The `source` fields
and separate historical table preserve these differences. Do not treat native
reward/cost values from different runtime definitions as automatically aligned.

The October 8 report authenticated accepted task summaries and checked table
consistency. This sharing release checks byte preservation and episode-row
copying. It does **not** independently rerun raw-trace scientific validation or
complete the separate archive-wide native reward/cost supplement.

Full time-series archives, weights, failed attempts, scheduler logs and
predictor-training collections are not included here. Penalty-selection
validation sweeps, predictor diagnostics and engineering probes are also
excluded from this outcome-metric release. The inventory records
about **37 GB compressed** of accepted outcome archives; that is a separate
storage tier, not the size needed to reuse these metrics. Structural integrity
does not establish physiological realism, forecast calibration or shielding
benefit for every configuration.

The result data in this folder is [MIT licensed](LICENSE). This does not change
the source-code license or the licenses of vendored dependencies.
