# Finance Autocomplete RL Training

## Purpose
Train a small LLM to complete text with financial data using RL + tool calls.

## Architecture
1. **Database**: Real Tiingo data → SQLite at `/tmp/financial_data.db`
   - Fetches from TWO Tiingo endpoints: `/statements` and `/daily`
   - 30 companies × 60+ metrics × multiple periods
   - Daily metrics aligned to statement periods (7-day lookback)
2. **Episodes**: Text input → Agent calls tools → Environment returns results → Agent returns completion
3. **Rewards**: LLM judge correctness score only (efficiency and completion penalties may be logged for analysis but do not affect training)
4. **Training**: ART framework on trajectories

## Key Files
- `database.py`: Tiingo data loading (statements + daily endpoints, exact field mapping)
- `environment.py`: Executes tools (get_metrics, get_tickers, get_value, calculate, return_answer)
- `agent.py`: Manages multi-turn tool calling conversations
- `rewards.py`: LLM-as-judge (same logic as main eval system)
- `rollout.py`: Generates training trajectories
- `server.py` + `index.html`: Test interface

## Database Improvements (Dec 2024)
- Added daily data endpoint for marketCap, peRatio, pbRatio, trailingPEG1Y
- Fixed metric mapping to use exact Tiingo dataCodes (netinc not netIncome)
- Expanded from 5 to 30 companies (added TSLA, NVDA, etc.)
- Added 40+ missing metrics from all statement types
- Percentages multiplied by 100 for readability (0.46 → 46%)

## Important
- NO sample data - requires real Tiingo API or fails
- Tool syntax: `get_value(metric="revenue", ticker="AAPL", period="2023FY")`
- Metric names MUST match Tiingo's exact dataCodes
- Episode ends with `return_answer(answer="...")`
- Self-contained - now matches parent project's data coverage

## Testing
Run `python server.py` → http://localhost:8000 to verify modules work.