# Methods, findings and limitations

## Task and data

An agent receives explicit symptoms and asks about additional symptoms from a fixed vocabulary. The simulator answers using patient slot dictionaries. The classifier predicts one of 12 disease labels. Local files contain 1,912 training, 239 development and 239 unlabeled test cases; vocabulary size is 118. These inspected counts supersede the original narrative's 2,374-case total.

The state is a vector in {-1, 0, 1}^118. Queries reveal a symptom's positive or negative value. Unlisted symptoms become negative when asked, which is a strong simulation assumption.

## Implementations

| Component | REINFORCE | PPO | Diagnosis-action exploration |
| --- | --- | --- | --- |
| Policy outputs | 118 queries | 118 queries | 118 queries + 12 diagnoses |
| Policy optimization | Discounted Monte Carlo policy gradient, entropy bonus | Clipped surrogate objective, learned value baseline, entropy bonus | REINFORCE |
| Classifier | MLP, 118 → 1024 → 1024 → 12 | Same classifier | Separate shallower MLP fallback |
| Stop rule | Disease-specific adaptive entropy threshold | Fixed entropy threshold | Diagnosis action or turn limit |
| Returns | Discounted rewards | Discounted returns, normalized return-minus-value advantage | Discounted rewards |

The shared inquiry policy uses hidden dimensions 5120 → 10240 → 5120. PPO's Critic uses 1024 → 512 → 1. The implementation does not use GAE. Policy learning rate is one fifth of the configured learning rate; classifier learning rate defaults to 1e-4. PPO defaults to four updates per rollout and clip epsilon 0.2. The unified launcher uses seed 514, maximum 15 steps and trials 1/2/3.

For the main variants, inquiry reward comprises a step cost, a reward for newly revealed slots, an extra positive-symptom reward, and positive relative entropy reduction. On a threshold-triggered diagnosis, correct predictions receive a net +mu diagnosis term and incorrect predictions -mu. The diagnosis-action variant uses ±10 diagnosis rewards and a different reward structure, so it is not an isolated action-space ablation.

## Recorded findings

The reports show 155/239 correct diagnoses for the baseline and 160/239 for PPO. The difference is five additional correct cases, about 2.10 percentage points. Mean question count changes from 9.4393 to 9.1255. Thyroiditis improves from 6/19 to 11/19; conjunctivitis falls from 15/21 to 13/21. Improvements are not uniform across classes.

These results support describing an observed development-set improvement in one recorded run. They do not establish statistical significance, superior generalization across datasets, or a clinical benefit. PPO's fixed threshold and the baseline's adaptive threshold confound a pure optimizer comparison.

## Diagnosis-action audit

The old environment reconstructed a fresh base environment inside every `initialize_state` call. Each call restarted its cursor at zero. With batch size 200 and 239 development patients, evaluation processed the first 200 patients twice, reporting 400 entries. Training similarly reused initial patients rather than traversing the split. Test predictions from that pipeline also cannot be assumed to correspond to the test set in order.

The repaired environment delegates state initialization on the persistent object. A synthetic regression check verifies a 2+1 batch traversal of three distinct patients. The historical report remains unchanged for traceability, but its 7.25% accuracy and inferred collapse cannot justify a causal conclusion about joint diagnosis actions. Retrain from scratch before interpreting that variant.

## Repository preparation changes

- Keep raw inputs in the original folder; create a separate presentation repository.
- Add a consistent launcher, explicit data checks, and checks that prevent missing diagnosis weights from silently producing random-model evaluation.
- Fix singleton batch shapes in the baseline agent, protect entropy normalization from division by zero, and copy adaptive-threshold snapshots when logging.
- Replace training-time validation with sample-weighted metrics and actual active-question counts.
- Correct `eval_dev.py`, which imported the PPO agent and wrote PPO output filenames despite being the baseline evaluator.
- Exclude PPO transitions collected after a patient's termination; use population standard deviation for singleton advantage normalization; pass through the configured discount factor.

These changes affect behavior. The historical numbers describe the original artifacts and must not be labeled as verified results of the current source.

## Remaining limitations

- No supplied checkpoints; no full training rerun in this cleanup.
- Exact data provenance and redistribution rights remain unresolved.
- REINFORCE uses nonzero discounted returns to filter trajectories rather than an explicit active mask; zero-return active transitions can be discarded.
- Policies sample stochastically at evaluation. Seeds improve repeatability but do not guarantee deterministic GPU results.
- Already observed symptoms remain selectable; repeated questions can waste turns.
- No text understanding, natural-language generation, realistic patient uncertainty or clinical validation.
- Historical plots use old metric conventions; the PPO training reward curve is a mean of step rewards, while the baseline records a mean episode return.

Useful follow-up experiments would match stopping rules, use multiple seeds, report confidence intervals and positive-only symptom recall, and rerun the diagnosis-action variant after the cursor fix.
