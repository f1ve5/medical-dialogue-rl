# Reproduction guide

## Environment status

The original report describes Python 3.10.20, PyTorch 2.11.0+cu128 and an RTX 5060 Laptop GPU with 32 GB system RAM. This is historical information from the report, not an independently reconstructed environment.

The new requirements list direct dependencies with compatibility ranges, replacing a full machine-specific freeze containing CUDA-local versions and unrelated packages. It is not a lockfile. CI targets Python 3.11 and runs environment regressions without PyTorch or private data. Full training compatibility must be checked on the selected PyTorch build.

## Data and local validation

1. Obtain permission to use the exact supplied split and place the five documented files in `dataset/`.
2. Run `python scripts/inspect_data.py` to validate structure and count cases.
3. Run `python -m unittest discover -s tests -v` for synthetic environment regressions.
4. Run `python -m compileall -q src scripts tests` for source syntax checks.

The inspection tool rejects class references in pickles. Training still assumes trusted local pickle files. Data is not downloaded automatically and is not tracked by Git.

## Training and evaluation

Use `python scripts/run.py MODEL MODE`, where MODEL is `reinforce`, `ppo` or `diagnosis`, and MODE is `train`, `evaluate` or `predict`.

```bash
# Pipeline smoke run; does not reproduce a 200-epoch experiment.
python scripts/run.py ppo train --epochs 1 --batch-size 8 --max-steps 2

# Full intended schedule, if hardware resources permit.
python scripts/run.py ppo train --epochs 200 --device gpu --seed 514
python scripts/run.py ppo evaluate --device gpu --seed 514
python scripts/run.py ppo predict --device gpu --seed 514
```

The policy alone has approximately 106 million parameters. Activations and Adam optimizer state add substantial memory requirements. Smaller batches reduce memory demand but change the training schedule and may change results. No runtime estimate is established.

Outputs go to ignored `log/`, `model_saved/` and `test_prediction*.txt`. Evaluation loads the best checkpoint where available. The supported launcher consistently passes threshold=1.0, mu=1.0 and nu=2.5, giving run suffixes such as `1.0_1.0_2.5_1`. Original files used both integer-like and floating-point identifiers; do not assume their filenames match new runs. Original pickle logs and metadata are deliberately not seeded into the output directory.

Baseline, PPO and diagnosis trials are 1, 2 and 3. Checkpoint loading must use matching model, run suffix and data vocabulary order. The launcher reports absent data or absent checkpoints; a checkpoint for another model does not satisfy the requested model's loader.

## Interpretation

The current code includes behavior fixes and does not claim exact reproduction of the old figures. Preserve new run configuration, dependency versions, seed, checkpoints and evaluation outputs together. Record multiple seeds and matched stopping conditions before making stronger comparative claims.

Test data has no labels. Prediction export can be checked for 239 rows and vocabulary membership, but test accuracy cannot be computed from these files.
