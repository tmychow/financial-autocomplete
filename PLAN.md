## Plan: Curriculum, Validation, Sampling, Metric Search, and Cleanup

- **Owner**: TBD
- **Repo**: `financial-autocomplete`
- **Scope**: Update training curriculum in `finance_rl.ipynb`, adjust generators in `synthetic.py`, tighten validation in `rollout.py`, improve matching in `environment.py`/`textmatch.py`, and remove redundant code.

### 1) Three-stage curriculum in the notebook (simpler schedule)

- [x] Update training steps and curriculum schedule in `finance_rl.ipynb`
  - [x] Set `NUM_STEPS` to 140
  - [x] Implement 3 phases with exact stage windows by `step` using `curriculum_stage`:
    - [x] Steps 0–59: weights = `[0.50, 0.30, 0.10, 0.10]`, `no_completion_ratio = 0.20`
    - [x] Steps 60–99: weights = `[0.40, 0.30, 0.15, 0.15]`, `no_completion_ratio = 0.25`
    - [x] Steps 100–139: weights = `[0.40, 0.20, 0.20, 0.20]`, `no_completion_ratio = 0.25`
  - [x] Keep `curriculum_stage` mapping in code (no direct weights passed from notebook)
  - [ ] Log the active phase, weights, and `no_completion_ratio` every step for traceability

- [x] Update `synthetic.generate_cases` stage mapping to the three-stage schedule (length 4 vectors)

- [ ] Make `rollout.generate_training_trajectories` forward explicit distribution settings (or allow the notebook to bypass it and call `conduct_rollouts` directly)
  - [ ] If keeping `generate_training_trajectories`, allow an override of distribution and disable its internal `curriculum_stage` logic

### 2) Remove multi-metric synthetic generation

- [x] In `synthetic.py`
  - [x] Remove `generate_multi_metric_calc_case` and `CALC_COMBOS`
  - [x] Update the `generators` list to only include: `generate_simple_case`, `generate_latest_case`, `generate_difference_case`, `generate_cross_ticker_difference_case`
  - [x] Update any default or stage weights to length 4 accordingly
  - [x] Delete/clean associated comments and dead code tied to multi-metric cases

- [x] Ensure no references remain
  - [x] Search usages of `generate_multi_metric_calc_case` and `CALC_COMBOS`
  - [x] Remove any tests or demo code referencing calc cases
  - [x] Keep the `calculate` tool in `environment.py` (it may be used in trajectories even if we no longer synthesize calc cases)

### 3) Validation rollouts always use terminal distribution

- [x] In `rollout.run_validation`
  - [x] Use `curriculum_stage=3` to select terminal distribution (weights = `[0.40, 0.20, 0.20, 0.20]`, `no_completion_ratio = 0.25`) per `synthetic.stage_to_weights`
  - [x] Use exact sampling (see section 5) to hit the distribution precisely per validation batch
  - [x] Log chosen validation holdouts (tickers/metrics/periods) for traceability

### 4) Validation/train split to prevent overfit

- [x] Define a deterministic validation holdout
  - [x] Select and fix sets: 5 tickers, 5 metrics, 5 periods + "latest"
  - [ ] Persist these selections in the training run state/log (and print in the notebook)

- [x] Constrain validation case generation to holdouts only
  - [x] Add params to `synthetic.generate_cases`: `allowed_tickers`, `allowed_metrics`, `allowed_periods` (a set that can include "latest")
  - [x] For each generator, ensure company/metric/period sampling respects these constraints (including cross-ticker cases)

- [x] Exclude holdout entities from training case generation
  - [x] Add params to `synthetic.generate_cases`: `excluded_tickers`, `excluded_metrics`, `excluded_periods`
  - [x] Ensure training still allows "latest" even if periods are otherwise constrained (per requirements)
  - [x] Add guardrails/fallbacks if constraints make a case impossible (skip and continue)

### 5) Exact-proportion sampling (no drift from randomness)

- [x] Implement deterministic, quota-based sampling in `synthetic.generate_cases`
  - [x] Compute the exact integer counts per generator type and `no_completion` given `num_cases` and provided weights/ratio
  - [x] Use rounding with a stable tie-breaker to ensure totals sum exactly to `num_cases`
  - [x] Generate cases in quota blocks (e.g., iterate generator types in fixed order to fill their counts) rather than `random.choices`
  - [x] Add retry limits and clear warnings if some quotas cannot be filled given constraints; fill shortfalls by the next available generator per a stable priority order
  - [x] No seed needed; quotas ensure realized proportions

### 6) Improve metric search and add "did you mean" suggestions

- [x] Robust similarity in `textmatch.py` (no exhaustive variant enumeration)
  - [x] Stopword-aware tokenization for signatures (drop connectors like "and", "&", "of", etc.)
  - [x] Combined similarity: raw difflib, signature difflib, and Jaccard over tokens
  - [x] Keep existing aliasing for codes/descriptions; no curated synonyms added

- [x] Enhance `_search` metric error handling in `environment.py`
  - [x] When invalid, return top-3 "did you mean" suggestions from alias keys
  - [ ] Optional: auto-correct when top candidate score is well above cutoff (configurable)

- [ ] Update agent handling (if needed)
  - [ ] Verify the agent surfaces suggestion text back to the model and encourage retrying

### 7) Redundant/duplicate code audit and removals

- [x] Remove unused or duplicate code
  - [x] `synthetic.py`: remove `generate_multi_metric_calc_case`, `CALC_COMBOS`
  - [x] `synthetic.py`: remove commented-out templates in `generate_simple_case`
  - [x] `synthetic.py`: remove unused `generate_training_data`
  - [x] `finance_rl.ipynb`: drop import of `generate_training_data`
  - [x] `synthetic.py`: drop unused imports

- [x] Grep for dead symbols and references
  - [x] Search for removed function names and constants
  - [x] Confirm `get_all_periods` and `build_metric_alias_map` have active call sites

- [x] Sanity checks
  - [x] Run lints and basic import checks
  - [x] Ensure sample generation and validation calls compile

### 8) Documentation and configs

- [x] Update `README.md`
  - [x] Explain repo purpose succinctly (tool-calling RL on real Tiingo data)
  - [x] Note three-stage curriculum, exact-proportion sampling, and validation holdout
  - [x] Mention removal of multi-metric synthetic cases
  - [x] Mention robust metric matching and "did you mean" behavior
  
- [x] Update `CLAUDE.md`
  - [x] Reflect curriculum, exact sampling, and validation setup
  - [x] Summarize tool usage, reward policy, and suggestions behavior

- [ ] Versioning and logs
  - [ ] Bump `MODEL_NAME` suffix (e.g., `_v23`) in `finance_rl.ipynb`
  - [ ] Ensure `TRAIN_LOG_PATH`/`VAL_LOG_PATH` record distribution and holdout configurations

