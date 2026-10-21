"""Replot the six recorded patient #009/#010 episodes; no simulator is needed."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

ARMS = ('none', 'static', 'predictive')
LABELS = ('No shield', 'Static shield', 'Predictive shield (penalty 5)')
COLORS = ('#BC433C', '#717D89', '#087F7D')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output-dir', type=Path, default=Path('output'), help='Output directory for both patients')
    parser.add_argument('--patient', choices=('009', '010'), help='Plot only one of the two patients')
    args = parser.parse_args()
    for patient in ((args.patient,) if args.patient else ('009', '010')):
        plot_one(args.root, patient, args.output_dir / ('patient' + patient))


def plot_one(root, patient, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    provenance = json.loads((root / 'PROVENANCE.json').read_text())
    pins = {item['condition']: item for item in provenance['episodes'] if item['case_id'] == 'p' + patient + '-d7-e83002'}
    observations, actions, metrics = [], [], []
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                            gridspec_kw={'height_ratios': [3.2, 1, 1, 1]}, constrained_layout=True)
    axes[0].axhspan(70, 180, color='#EDF5EF')
    for y in (70, 180):
        axes[0].axhline(y, color='#AEC8B4', lw=.8, ls='--')
    for i, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
        directory = root / ('episodes/p' + patient + '-d7') / arm
        metadata = json.loads((directory / 'metadata.json').read_text())
        trace = directory / 'source/trace.npz'
        if hashlib.sha256(trace.read_bytes()).hexdigest() != pins[arm]['trace_sha256']:
            raise ValueError('Original NPZ checksum mismatch: ' + arm)
        with np.load(trace, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        obs = arrays['observations']
        n = len(arrays['rewards'])
        if obs.shape != (2017, 14) or n != 2016 or metadata['patient_name'] != 'adolescent#' + patient:
            raise ValueError('Unexpected patient or recorded horizon')
        post = obs[1:, 0]
        statistics = dict(condition=arm, transitions=n, coverage_percent=100.0,
                          tir_percent=float(100 * np.mean((post >= 70) & (post <= 180))),
                          below54_percent=float(100 * np.mean(post < 54)),
                          terminated=bool(arrays['terminated'][-1]), truncated=bool(arrays['truncated'][-1]))
        metrics.append(statistics)
        axes[0].plot(np.arange(n + 1) * 5 / 60, obs[:, 0], color=color, lw=1.3,
                     label=f'{label}: TIR {statistics["tir_percent"]:.2f}%')
        for k, values in enumerate(obs):
            observations.append(dict(condition=arm, observation_minute=5*k, is_reset=k == 0,
                                     **dict(zip(metadata['observation_features'], values.tolist()))))
        requested, executed, accepted = arrays['recommended_actions'], arrays['executed_actions'], arrays['accepted']
        ax = axes[i+1]
        for j, (name, offset) in enumerate((('bolus', 0), ('meal', 6))):
            for ok, marker in ((True, 'o'), (False, 'x')):
                mask = (requested[:, j] > 0) & (accepted[:, j].astype(bool) == ok)
                ax.scatter(np.arange(n)[mask] * 5 / 60, requested[mask, j] + offset,
                           color=color, marker=marker, s=10)
        ax.set_yticks([2.5, 8.5], ['Bolus\nlevels 1–4', 'Meal\nlevels 1–4'])
        ax.set_ylim(.3, 10.7)
        ax.text(.995, .92, label, transform=ax.transAxes, ha='right', va='top', fontsize=9)
        for k in range(n):
            actions.append(dict(condition=arm, decision_minute=k*5, observation_minute=(k+1)*5,
                                bolus_requested_level=int(requested[k, 0]), meal_requested_level=int(requested[k, 1]),
                                bolus_executed_level=int(executed[k, 0]), meal_executed_level=int(executed[k, 1]),
                                bolus_accepted=bool(accepted[k, 0]), meal_accepted=bool(accepted[k, 1]),
                                reward=float(arrays['rewards'][k]), cost=float(arrays['costs'][k]),
                                terminated=bool(arrays['terminated'][k]), truncated=bool(arrays['truncated'][k]),
                                forecast_available=bool(arrays['forecast_available'][k])))
    axes[0].set_ylabel('CGM (mg/dL)')
    axes[0].legend(fontsize=9)
    axes[0].set_title(f'PPO-Lag · T2D no pump · adolescent #{patient} · seven recorded days\n'
                      'Policy seed 101 / epoch 976 · deterministic · evaluation seed 83002 · predictor seed 3103')
    axes[-1].set_xlim(0, 168)
    axes[-1].set_xlabel('Simulated time (hours); ○ accepted request, × declined request')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#EEF0F2', lw=.6)
    fig.supxlabel('Benefit-selected illustration, not a cohort estimate. Recorded samples; no smoothing or interpolation. '
                  'Recommendation levels are not physical doses.', fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.png', '.pdf'):
        fig.savefig(output.with_suffix(extension), dpi=160)
    plt.close(fig)
    for suffix, rows in (('.observations.csv', observations), ('.actions.csv', actions), ('.metrics.csv', metrics)):
        with output.with_suffix(suffix).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
