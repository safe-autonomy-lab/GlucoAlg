# Render a trained policy

This workflow loads an existing categorical policy checkpoint, restores its
saved observation normalization, samples actions in GlucoSim, and exports a
GIF with glucose, bolus/meal requests, and simulator acceptance decisions.
No training or runtime shield is added.

## Install

Use the same simulator revision as the training run. For the verified Hugging
Face `rerun-20261008` release, the model manifest specifies GlucoSim commit
`ddd674e35a32a3a6406a0c2d5e7bf3c72f56dfec`. The commands below use new checkouts
and a new CPU environment; existing training environments need not be changed.

```bash
mkdir gluco-demo
cd gluco-demo
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu

git clone https://github.com/safe-autonomy-lab/GlucoSim.git
git clone https://github.com/safe-autonomy-lab/GlucoAlg.git
git -C GlucoSim checkout ddd674e35a32a3a6406a0c2d5e7bf3c72f56dfec
python -m pip install -e ./GlucoSim -e './GlucoAlg[render]'
```

The `render` extra declares Pillow and Hugging Face Hub and selects Gymnasium
1.3.x, which is compatible with the pinned simulator release. Gymnasium 1.4's
type-variable defaults cause that release to fail during import. FFmpeg is not required.
GlucoAlg's modified OmniSafe is bundled; do not install another upstream copy.

## A student's own checkpoint

The trainer saves `config.json`, `runtime.json`, `progress.csv`, and a
`torch_save/epoch-N.pt` directory under each timestamped run. Choose a
checkpoint and configuration from the **same run**. The epoch depends on
training duration; `976` below is an example, not a required filename.

```bash
glucoalg-demo \
  --checkpoint /absolute/path/to/run/torch_save/epoch-976.pt \
  --config /absolute/path/to/run/config.json \
  --simulator-root ./GlucoSim \
  --patient 'adolescent#002' \
  --eval-seed 43 \
  --days 1 \
  --output-dir ./student-demo
```

Algorithm, disease type, and policy training seed are read from `config.json`.
Select the intended evaluation cohort/patient explicitly. For example, use
`'adult#002'` for an adult evaluation. The checkpoint's `pi` state supplies the
actor weights; `obs_normalizer`, when present, supplies the frozen input
transform. The loader rejects incompatible observation widths and actor weights.

## An uploaded Hugging Face policy

The example below uses a checked release of
[`safe-diabetes-t2d-adolescent-cpo`](https://huggingface.co/safe-diabetes-benchmark/safe-diabetes-t2d-adolescent-cpo).
Its three policy training seeds are **100, 101, 102**, endpoint **epoch 976**.

```bash
glucoalg-demo \
  --hf-repo safe-diabetes-benchmark/safe-diabetes-t2d-adolescent-cpo \
  --revision 557192c4db96edf90f3b9e239f4ae5ebdc7781df \
  --training-seed 100 \
  --simulator-root ./GlucoSim \
  --patient 'adolescent#002' \
  --eval-seed 43 \
  --days 1 \
  --output-dir ./hf-cpo-demo
```

Only the selected policy checkpoint, matching config, and release manifest
are downloaded. The revision is resolved once and pinned for every download;
both checkpoint and config hashes are verified against the release manifest.
Use `--cache-dir /path/to/cache` to choose the download cache.

Exact paths for seed 100:

```text
releases/rerun-20261008/checkpoints/seed100/epoch-976.pt
releases/rerun-20261008/config/seed100/config.json
```

The checked adolescent PPO-Lag release uses the same layout at repository
`safe-diabetes-benchmark/safe-diabetes-t2d-adolescent-ppolag`, revision
`ea45fc5a4ae288d04330a456b7a6d6c003247ccf`. Change both `--hf-repo` and
`--revision` when switching repositories. Historical September seeds 0/1/2
and epoch 2441 belong to older pinned releases; do not mix them with this one.

## Outputs and re-rendering

The output directory must be new. A successful invocation creates:

```text
hf-cpo-demo/
  policy_source.json
  evaluation/
    EVAL_MANIFEST.json
    adolescent#002/
      episode_0_controller.csv
      detailed_results.csv
      summary_results.csv
  episode.gif
  episode.png
  episode.render.json
  episode.gif.lock
```

The manifest records checkpoint/config hashes, the actual simulator source,
stochastic action mode, evaluation seed, episode length and horizon coverage.
The GIF receipt records the input CSV hash, output duration and glucose axes.
The lock file reserves the output name, including after a partial/failed render.
Choose a new output name for a retry; existing results are never overwritten.

To change playback speed or title without rerunning the policy:

```bash
glucoalg-render \
  --trace './hf-cpo-demo/evaluation/adolescent#002/episode_0_controller.csv' \
  --output ./hf-cpo-demo/episode-slower.gif \
  --seconds 30 \
  --title 'CPO — T2D adolescent #002'
```

`python -m glucoalg.demo` and `python -m glucoalg.rendering` are equivalent
module entry points. To call the recorder and renderer directly in Python:

```python
from pathlib import Path
from glucoalg.eval_grid import run_grid
from glucoalg.rendering import render

run_grid(
    checkpoint=Path('/absolute/run/torch_save/epoch-976.pt'),
    config=Path('/absolute/run/config.json'),
    output_dir=Path('my-evaluation'),
    patient_type='t2d', patients=['adolescent#002'],
    episodes=1, horizon_days=1, eval_seed_base=43,
    action_mode='stochastic', glucose_source='post-step',
    simulator_root=Path('GlucoSim'), save_traces=True,
)
render(
    Path('my-evaluation/adolescent#002/episode_0_controller.csv'),
    Path('my-episode.gif'),
)
```

## Reading the animation

- `--days 7` records a week instead of a day. Playback is approximately
  20 seconds unless `--seconds` is supplied; GIF timing is quantized to 10 ms.
- All recorded glucose samples remain in the growing line. Frames are
  subsampled for playback; persistent event lanes retain every nonzero request.
  The top label shows the most recent request visible at each frame.
- Requests occur five minutes before their associated post-step observation.
  Keep `EVAL_MANIFEST.json` alongside the CSV so this alignment is checked.
  A detached CSV needs explicit `--glucose-source post-step`, after its source
  protocol has been confirmed. Legacy pre-action CSVs are rejected.
- Recommendations and acceptance are separate from delivered physical doses.
  The GIF labels bolus/meal **levels**, not delivered insulin or meal amounts.
- Glucose axes include the whole recorded range. Early endings are drawn only
  for the observed duration; inspect horizon coverage in the manifest.

This is an individual simulator demonstration. It does not reproduce the
model card's multi-patient, multi-seed seven-day aggregate results. Use the
existing [predictive-shield workflow](predictive-shield.md) for interventions
requiring a predictor artifact; these demo commands are unshielded.

The older `eval_run.py --render` helper has different data requirements and
can finish without producing a GIF. Use the commands on this page for the
checkpoint animation workflow.
