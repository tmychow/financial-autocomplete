# financial-autocomplete

Co-designing a financial data autocomplete product with the reinforcement learning finetuning of a cheap, fast model to use tool calls.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

1. **Tiingo API Key** (Required)
   - Get a free key at: https://api.tiingo.com/account/api/token
   - Enable the "Fundamental Data" add-on (also free)

2. **OpenAI API Key** (Required)
   - Get from: https://platform.openai.com/api-keys

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Database Setup

On first run, the database loads financial data from Tiingo statements and daily endpoints. It covers dozens of companies and a broad set of income, balance sheet, cash flow, and market metrics. Data are stored locally in `financial_data.db`.

### Test Server

Run the test server to verify everything works:

```bash
python server.py
```

Open http://localhost:8000 to:
- Browse financial data in the database
- Run quick model evaluations
- View trajectories

### Training

Open `finance_rl.ipynb` to run training. The notebook orchestrates:
- Curriculum learning (three stages) with exact-proportion sampling per batch
- Tool-based rollouts and LLM-as-judge rewards
- Periodic validation against a benchmark model

Reward combines judge correctness with tool-use/coverage signals (ticker/metric/period correctness, lookup coverage, used search).

Validation uses a fixed holdout: a small, deterministic subset of tickers, metrics, and periods (plus “latest”), and a terminal task distribution. This avoids overfitting the validation set during training.

## Files

- `database.py`: Tiingo loaders (statements + daily) and query utilities
- `synthetic.py`: Synthetic case generation with curriculum-aware, exact-proportion sampling
- `environment.py`: Tool execution (search, calculate, return_answer) with tolerant normalization and “did you mean” metric suggestions
- `agent.py`: Multi-turn agent driving tool calls
- `rewards.py`: Judge-based rewards + tool-use/coverage signals
- `rollout.py`: Rollout execution and validation
- `server.py` + `index.html`: Minimal web UI (OpenAI and local Ollama supported)

## Data Coverage

Data use Tiingo’s exact field names across statements and daily endpoints (e.g., `netinc`, `ebitda`, `freeCashFlow`, `marketCap`).

## TODO

- [ ] Track gradient norm and KL divergence each step; pause training if they explode
- [ ] Keep more than the most recent checkpoint; ensure a checkpoint from last 5 steps is always retained
- [ ] Convert to a model which can be used locally
- [ ] Improve plots