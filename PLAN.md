## Implementation Plan: Tool Simplification, Flexible get_value, Enhanced Rewards

### Goals
- Replace current tool set with only three tools: `get_value`, `calculate`, `return_answer`.
- Make `get_value` flexible and forgiving:
  - Accept company names or partials (e.g., "Apple" → `AAPL`).
  - Accept natural-language metric names/synonyms (e.g., "net income" → `netinc`, "price to earnings" → `peRatio`).
  - Accept loose period formats (e.g., "2023", "FY2023", "Q4 2023", "latest").
- Extend rewards beyond LLM-as-judge to include:
  - Whether the model used `get_value` at all (for data-required cases).
  - Whether `get_value` calls covered the correct metric/ticker/periods per the synthetic case.
- Extend synthetic cases to save ground-truth lookup metadata enabling the new reward checks.

### Non-Goals
- No schema/data changes to the SQLite DB (we use current tables/functions).
- No change to numerical formatting of ground truths produced by `synthetic.py`.

---

## Task Checklist (trackable)

### environment.py
- [x] Remove `ToolName.GET_METRICS`, `ToolName.GET_TICKERS` from enum
- [x] Remove `_get_metrics` and `_get_tickers` methods
- [x] Remove corresponding branches in `execute_tool`
- [x] Update `parse_tool_calls_from_response` `tool_signatures` to only `get_value`, `calculate`, `return_answer`
- [x] Remove parser no-arg handling for `get_metrics`/`get_tickers`
- [x] Implement alias-building on init:
  - [x] Build ticker alias map from `get_tickers_with_names()`: ticker, company name (case-insensitive), substring matches; manual aliases (e.g., alphabet/google → `GOOGL`, facebook/meta → `META`)
  - [x] Build metric alias map from `get_all_metrics()` names/descriptions + manual synonyms (see Alias Guidance below)
- [x] Implement `_normalize_ticker(raw: str) -> Optional[str]`
- [x] Implement `_normalize_metric(raw: str) -> Optional[str]`
- [x] Implement `_normalize_period(raw: str, ticker: str, metric: str) -> Optional[str]` with patterns for FY, Q, and `latest`
- [x] Update `_get_value` to normalize inputs, resolve `latest` via `get_latest_period`, fetch via `get_financial_value`, and return result including normalized `ticker`, `metric`, and concrete `period`
- [x] Update self-test block: remove metrics/tickers tests; add normalization examples

### agent.py
- [x] Update `SYSTEM_PROMPT` to list only the three tools
- [x] Emphasize flexible `get_value` (company names and natural metric terms); include 1–2 examples
- [x] Remove instruction to call `get_metrics`/`get_tickers` first
- [x] Ensure conversation loop unchanged; finalization still via `return_answer`

### synthetic.py
- [x] Augment each case generator to include `metadata` with ground-truth lookups
  - [x] `generate_simple_case`: 1 required lookup `{ticker, metric, period}`
  - [x] `generate_latest_case`: 1 required lookup with `period: "latest"` (and the resolved latest period is known internally)
  - [x] `generate_difference_case`: 2 required lookups for the same ticker/metric across two periods
  - [x] `generate_cross_ticker_difference_case`: 2 required lookups for two tickers, same metric, same period
  - [x] `generate_multi_metric_calc_case`: 2 required lookups for same ticker, two metrics, same period
  - [x] `generate_cagr_case`: 2 required lookups for same ticker/metric at start/end FY; include `calc: {operation: "CAGR", duration: years}`
  - [x] `generate_no_completion_case`: `required_lookups: []`, `type: "no_completion"`
- [x] Ensure `generate_cases` propagates `metadata` alongside `input` and `ground_truth`
- [x] Ensure `generate_training_data` remains compatible

### rewards.py
- [x] Extend function signature to accept `tool_calls` and `case_metadata`
  - `calculate_reward(prediction, ground_truth, episode_info, tool_calls=None, case_metadata=None, use_judge=True, judge_model=None)`
- [x] Keep LLM-as-judge correctness as main component
- [x] Add tool usage bonus: +w_used_value if any tool call with `tool == "get_value"` (ignored for `no_completion`)
- [x] Add lookup coverage bonus: compare `get_value` results’ `{ticker, metric, period}` against `case_metadata.required_lookups`
  - [x] Handle `latest` by matching the concrete `period` returned in tool results (since `_get_value` fills it in)
  - [x] Coverage = matched_required / total_required
- [x] Combine: `total_reward = w_judge * correctness + w_used_value * used_flag + w_coverage * coverage`
- [x] Make weights configurable via env vars with sensible defaults
- [x] Update `calculate_batch_rewards` to pass through new args

### rollout.py
- [x] Pass `tool_calls` and `case_metadata` to `calculate_reward`
- [x] Optionally record sub-scores in `metrics` (e.g., coverage, used_get_value)

### server.py
- [x] Update reward call to pass `tool_calls` and `case_metadata`
- [x] Optionally return sub-scores and reasons in API response

### Docs and Examples
- [x] README.md: update tools list, examples, and reward description
- [x] CLAUDE.md: update tool references and example syntax
- [ ] finance_rl.ipynb: update narrative/tool descriptions; ensure any sample strings referencing removed tools are updated

### Optional/Quality
- [ ] Add simple unit tests or assertions for `_normalize_ticker`, `_normalize_metric`, `_normalize_period`
- [ ] Add logging for normalization fallbacks to aid debugging

---

## Alias Guidance (initial seed for normalization)

### Ticker Aliases
- Exact ticker (e.g., `AAPL`)
- Exact company name (case-insensitive), substring/prefix matches
- Manual:
  - "google", "alphabet" → `GOOGL`
  - "facebook", "meta" → `META`
  - Common punctuation-insensitive forms (e.g., remove Inc., Corp., N.V.)

### Metric Aliases (examples)
- Income statement: "revenue" → `revenue`; "net income" → `netinc`; "eps" → `eps`; "diluted eps"/"eps diluted" → `epsDil`; "ebitda" → `ebitda`; "ebit" → `ebit`; "r&d" → `rnd`; "sg&a" → `sga`; "operating income" → `opinc`; "cost of revenue" → `costRev`; "gross profit" → `grossProfit`; "operating expenses" → `opex`.
- Balance sheet: "assets" → `assets`; "ppe"/"pp&e" → `ppeq`; "accounts receivable" → `acctRec`; "cash and equivalents" → `cashAndEq`; "current assets" → `assetsCurrent`; "inventory" → `inventory`; "accounts payable" → `acctPay`; "total liabilities" → `totalLiabilities`; "equity" → `equity`; "deferred revenue" → `deferredRev`; "debt" → `debt`.
- Cash flow: "free cash flow" → `freeCashFlow`; "dividends paid" → `payDiv`; "depreciation and amortization" → `depamor`; "capex" → `capex`; "net cash from ops" → `ncfo`; "net cash from invest" → `ncfi`; "net cash from financing" → `ncff`; "stock-based comp" → `sbcomp`.
- Overview/ratios: "revenue qoq"/"revenue growth qoq" → `revenueQoQ`; "gross margin" → `grossMargin`; "profit margin" → `profitMargin`; "debt to equity" → `debtEquity`; "long-term debt to equity" → `longTermDebtEquity`; "roe"/"return on equity" → `roe`; "roa"/"return on assets" → `roa`; "book value per share" → `bvps`; "book value" → `bookVal`; "revenue per share" → `rps`.
- Market/daily: "market cap" → `marketCap`; "pe"/"p-e"/"price to earnings" → `peRatio`; "pb"/"price to book" → `pbRatio`; "peg" → `trailingPEG1Y`.

### Period Normalization
- "latest", "most recent" → latest period via `get_latest_period`
- Fiscal year: `YYYY` or `FYYYYY` or `YYYYFY` → `YYYYFY`
- Quarter: any of `Qn YYYY`, `YYYY Qn`, `YYYYQn` → `YYYYQn` (n in {1,2,3,4})
- Disambiguate by intersecting with `get_available_periods(ticker, metric)` when needed

---

## Reward Weights (defaults; configurable via env)
- Judge correctness: `W_JUDGE = 1.0`
- Used `get_value` flag: `W_USED_GET_VALUE = 0.1`
- Lookup coverage: `W_COVERAGE = 0.2`
- For `type == "no_completion"`, treat `W_USED_GET_VALUE = 0` and `W_COVERAGE = 0`.

---

## Data Structures

### Tool call log entry (already emitted by `environment.py`)
```json
{
  "tool": "get_value" | "calculate" | "return_answer",
  "arguments": { ... },
  "result": { ... },
  "timestamp": "ISO-8601"
}
```

For `get_value`, `result` should include: `{"value": number, "unit": str, "ticker": str, "metric": str, "period": str}`.

### Synthetic case with metadata
```json
{
  "input": str,
  "ground_truth": str,
  "metadata": {
    "type": "simple" | "latest" | "difference" | "cross_ticker_difference" | "calc" | "cagr" | "no_completion",
    "required_lookups": [
      {"ticker": "AAPL", "metric": "revenue", "period": "2023FY"}
      // additional lookups as needed
    ],
    "calc": {"operation": "add|subtract|multiply|divide|CAGR", "duration": int|null}
  }
}
```

---

## Acceptance Criteria
- Agent successfully completes prompts like "Apple revenue in 2023 was" using `get_value(metric="revenue", ticker="Apple", period="2023FY")` thanks to normalization.
- Synthetic cases include `metadata.required_lookups`; reward computation uses it to award used-`get_value` and coverage bonuses.
- No code path references `get_metrics`/`get_tickers` in code, docs, or examples.
- Build runs and demo server endpoint still function; no new linter errors.

---

## Next Up
- Implement `environment.py` changes (enum removal, parser update, normalization, `_get_value` expansion). Then smoke test with manual calls.
- Add `metadata` to generators in `synthetic.py` and validate a sample of cases.
- Update `rewards.py` and wire through `rollout.py` and `server.py`.
- Update `agent.py` prompt and docs.

## Tracking Log
- [ ] 2025-08-11: Plan created and checked in as `PLAN.md`.


