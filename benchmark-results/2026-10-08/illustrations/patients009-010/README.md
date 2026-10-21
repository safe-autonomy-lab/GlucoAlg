# PPO-Lag patients #009 and #010: recorded shield comparison

[Download the complete 4.7 MB ZIP](https://huggingface.co/datasets/safe-diabetes-benchmark/transition_datasets/resolve/patients009-010-shield-comparison-2026-10-21/illustrations/ppolag-t2d-no-pump-adolescents009-010/ppolag-patients009-010-shield-comparison.zip?download=true). It includes the six recordings, CSVs, figures and code.

Only **two patients and six matched seven-day episodes** are included:
T2D without pump, adolescents #009 and #010, PPO-Lag policy seed 101 / epoch 976,
deterministic actions, environment/action seed 83002, exploration seed 183002.
The predictive arm uses predictor seed 3103 and absolute forecast penalty 5.
All six recordings completed 2,016 transitions, with no early termination.

| Patient | Mode | TIR 70–180 mg/dL (%) | CGM below 54 mg/dL (%) |
|---|---|---:|---:|
| #009 | No shield | 43.20 | 23.36 |
| #009 | Static shield | 49.16 | 8.78 |
| #009 | Predictive shield, penalty 5 | 77.58 | 0.64 |
| #010 | No shield | 57.84 | 14.93 |
| #010 | Static shield | 63.34 | 4.32 |
| #010 | Predictive shield, penalty 5 | 89.93 | 1.34 |

These are the previously highlighted examples selected for visible predictive
shield improvements. They are illustrations selected using outcomes, not cohort
averages. The penalty itself was selected earlier using separate seen-patient
validation. TIR includes both boundaries; below 54 is strictly below 54.

## Plot locally

Unzip the download, enter its folder, and run:

```bash
python -m pip install numpy matplotlib pandas pandas
python code/plot_trajectories.py --output-dir output
```

This produces PNG/PDF figures, reset-inclusive observation CSV, action/outcome
CSV, and a three-row metrics CSV per patient. Those outputs are also supplied in `output/`
so you can read the CSVs directly in your preferred plotting software.
No simulator, trained model, JAX, GPU, or Slurm is required.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output/patient009.observations.csv')
for mode, rows in df.groupby('condition', sort=False):
    plt.plot(rows.observation_minute / 60, rows.cgm, label=mode)
plt.xlabel('Simulated hours')
plt.ylabel('CGM (mg/dL)')
plt.legend()
plt.savefig('my-comparison.png', dpi=200)
```

To recreate the original synchronized GIF/MP4 animation locally:

```bash
python -m pip install pillow imageio-ffmpeg
python code/render_comparison.py --episode episodes/p009-d7 --output new-animation
```

Use `episodes/p010-d7` to render patient #010. The animation output directory must be new. Animation files are generated
locally; the download contains the data and rendering code.

## Included data

Each `episodes/{p009-d7,p010-d7}/{none,static,predictive}/` folder contains:

- `source/trace.npz`: all 25 original arrays, with unchanged bytes/dtypes/shapes.
- `source/report.json`, `source/decisions.json`: original outcomes and shield decisions.
- `source/task-spec.json`: original policy/predictor hashes, configuration and seeds.
- `rollout.json`: convenient recorded per-transition observations/actions.
- `metadata.json`: identity, feature names, metrics and original source hashes.

Task specs retain their original case lists for provenance; the archive contains
actual trajectory data for **patients #009 and #010 only**. Historical server paths in
JSON are provenance strings and are not needed for local plotting.
`PROVENANCE.json` and `SHA256SUMS` identify the original recordings and included
files. `DATA_DICTIONARY.md` explains timing, arrays and units.

CGM observations are recorded every five minutes. The reset is minute 0;
action `k` is requested at minute `5*k` and precedes observation `k+1` at minute
`5*(k+1)`. TIR uses the 2,016 post-step samples, excluding reset. Plotting adds no
smoothed or interpolated samples. Accepted recommendation levels are not
physical insulin units or carbohydrate grams. Full predicted future curves and
physical delivery histories were not recorded; original forecast diagnostics
and all retained decision records are included.

The synthetic data and included helper scripts are [MIT licensed](LICENSE).
This does not change the license of GlucoAlg, GlucoSim or their dependencies.
