# Finance Autocomplete RL Training

## Purpose
Train a small LLM to complete financial text using tool calls (search, calculate) and reinforcement learning on trajectories.

## Architecture
1. **Database**: Tiingo statements + daily endpoints → SQLite (`financial_data.db`)
   - Daily metrics are aligned to statement periods (7-day lookback)
2. **Episodes**: Input prefix → Agent calls tools → Environment returns results → Agent returns completion
3. **Rewards**: LLM-as-judge correctness + tool-use signals (used search, required lookup coverage, ticker/metric/period correctness)
4. **Training**: ART trajectories with exact-proportion sampling

## Key Files
- `database.py`: Load/query Tiingo data (statements + daily)
- `environment.py`: Execute tools; normalize inputs; provide "did you mean" metric suggestions
- `agent.py`: Multi-turn tool-calling agent
- `rewards.py`: Judge-based reward + tool-use and coverage signals
- `rollout.py`: Training rollouts and validation
- `server.py` + `index.html`: Minimal test UI (OpenAI + local Ollama)

## Data Coverage
- Added daily data endpoint for marketCap, peRatio, pbRatio, trailingPEG1Y
- Fixed metric mapping to use exact Tiingo dataCodes (netinc not netIncome)
- Expanded from 5 to 30 companies (added TSLA, NVDA, etc.)
- Added 40+ missing metrics from all statement types
- Percentages multiplied by 100 for readability (0.46 → 46%)

## Notes
- Real Tiingo data required; no mock dataset
- Tool syntax: `search(metric="revenue", ticker="Apple", period="2023")`
- Metric names map to Tiingo dataCodes (tolerant matching with suggestions)
- Episode ends with `return_answer(answer="...")`

## Testing
Run `python server.py` and open http://localhost:8000.