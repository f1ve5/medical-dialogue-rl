# Local verification record

Date: 2026-10-22.

Local check environment: Python 3.13.5, PyTorch 2.12.0.dev20260408+cu128, NumPy 2.3.5 and SciPy 1.17.1. This existing development build was used only for the stated local checks; it is not a recommended training lockfile or the historical training environment.

| Check | Outcome |
| --- | --- |
| `python scripts/inspect_data.py` | Passed: 1,912 train / 239 dev / 239 unlabeled test, 12 diseases, 118 symptoms |
| `python -m unittest discover -s tests -v` | 7 passed, including the two optional PyTorch tests |
| `python -m compileall -q src scripts tests` | Passed |
| Local Markdown link resolution | Passed |
| Git ignored-file inspection | Dataset payloads and Python caches excluded |
| Evaluate without model weights | Expected explicit failure; no evaluation generated |

The agent tests replace the large networks with tiny networks to check singleton inference dimensions and finite PPO updates. Environment tests use synthetic patients to check cursor progression, partial batches, weighted accuracy, terminal-state freezing, diagnosis rewards and missing test labels.

No full training, real-checkpoint inference, GPU training, historical-metric reproduction or hosted GitHub Actions run was performed. Compatibility ranges in requirements are not all covered by this local check. CI installs no PyTorch, so the two optional agent tests are skipped there unless that dependency is added.
