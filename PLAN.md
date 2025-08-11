## Trajectory Logging and Viewer Plan

### Goal
Add transparent, downloadable trajectory logging during training/validation (JSONL), and a simple client-only viewer in `index.html` to inspect conversations, tool calls, judge reasoning, and rewards.

### Outcomes
- [x] JSONL logs produced during training and validation without modifying ART internals
- [x] Each line captures: request, conversation, tool calls (with results), judge reasoning/score, reward breakdown, timing, and identifiers
- [x] Frontend viewer loads a JSONL file and lets you browse/filter episodes

---

### Phase 1 — Logging infrastructure (backend)
- [x] Add JSONL helpers in `rollout.py`
  - [x] `_append_jsonl(path: str, obj: dict)` ensures directory and appends
  - [x] `_make_rollout_log_record(...)` creates a complete, stable record
- [x] Extend `run_single_rollout` to build and return `log_record`
- [x] Extend `conduct_rollouts(..., log_path: Optional[str])` and append one line per successful rollout
- [x] Extend `run_validation(..., log_path: Optional[str])` and append one line per validation rollout (include benchmark reward)
- [x] Ensure record contains: `timestamp`, `step`, `rollout_id`, `input`, `ground_truth`, `prediction`, `conversation`, `tool_calls`, `episode_info`, `reward_info` (incl. judge reasoning/score), `latency_sec`, `model`, `judge_model`, `is_validation`, `benchmark_reward`
- [ ] Quick sanity check: single rollout writes exactly one JSON line

### Phase 2 — Notebook wiring (`finance_rl.ipynb`)
- [x] Add logging config in a config cell:
  - [x] `LOG_DIR=./logs`, ensure directory exists
  - [x] `RUN_TS` timestamp
  - [x] `TRAIN_LOG_PATH`, `VAL_LOG_PATH`
  - [x] Print where logs will be written
- [x] Pass `log_path=TRAIN_LOG_PATH` to `conduct_rollouts`
- [x] Pass `log_path=VAL_LOG_PATH` to `run_validation`
- [ ] Optional: log the single pre-training rollout

### Phase 3 — Frontend viewer (`index.html`)
- [x] Add a "Trajectory Log Viewer" section
- [x] File input to load a JSONL file from disk (client-side only)
- [x] Basic filters: only correct, only incorrect, only validation
- [x] Show summary counts
- [x] Render per-episode card with:
  - [x] Step/rollout id, reward, judge score, correct/incorrect indicator
  - [x] Input / ground truth / prediction
  - [x] Judge reasoning
  - [x] Tool calls in a small table (tool, args, result)
  - [x] Expandable conversation view
- [ ] QA using a sample log file

### Phase 4 — Nice-to-haves (optional)
- [ ] Size guard: truncate very large conversations when logging
- [ ] Optionally include `messages_and_choices` for deeper debugging
- [ ] Backend endpoint to list recent logs and serve them to the UI
- [ ] Auto-refresh or streaming viewer

### Acceptance Criteria
- [ ] After one training step, `logs/train_<timestamp>.jsonl` contains one line per successful rollout
- [ ] After one validation run, `logs/validation_<timestamp>.jsonl` contains one line per validation case
- [x] Each record includes the fields listed in Phase 1
- [x] The viewer loads a JSONL log and allows interactive inspection and filtering

### How to Run (once implemented)
1. Run training in `finance_rl.ipynb`.
2. Inspect generated logs in `./logs/`.
3. Open `index.html` in a browser, load a JSONL file, and browse records.

### Notes / Risks
- No WandB dependency; logs are appended locally and are ART-agnostic
- Be mindful of potential sensitive data in logs if sharing externally
- Log files can grow; rotate by run timestamp (already planned)


