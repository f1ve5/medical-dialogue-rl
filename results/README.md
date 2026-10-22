# Historical results and provenance

`historical/` contains unchanged aggregate reports copied from the supplied project's `code/log/` folder. `assets/` contains the supplied figures. These are records of earlier experiments, not outputs of the repaired source.

| Report | Recorded accuracy | Evidence / status |
| --- | ---: | --- |
| [REINFORCE](historical/dev_evaluation_report.txt) | 0.648536 | 155 correct / 239; class counts sum to 239 |
| [PPO](historical/dev_evaluation_report_ppo.txt) | 0.6695 | 160 correct / 239 from per-class counts |
| [Diagnosis actions](historical/dev_evaluation_report_diag_1.0_1.0_2.5_3.txt) | 0.072500 | 29 / 400 processed entries; repeated first-batch bug; invalid full-dev estimate |

The original training logs and test predictions remain in the untouched source folder. They are not bundled into the public presentation: raw pickle logs are unnecessary for reading these reports and the diagnosis predictions share the batching problem. No model checkpoint was supplied.

“Recall” in the existing evaluators divides newly discovered **positive** implicit symptoms by the number of **all** implicit slots for each patient, then averages over patients (zero-denominator patients contribute zero). It is not standard positive-symptom recall, disease recall, or macro-F1. The original metric name is preserved in the table for traceability.

Baseline and PPO use different stopping thresholds (adaptive versus fixed). Training checkpoint selection also used unweighted batch averages before the cleanup. Corrected source can produce different results. Multiple seeds, matching stop rules and fresh checkpoints are necessary for a controlled algorithm comparison.
