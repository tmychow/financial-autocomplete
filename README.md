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

The database automatically loads financial data from Tiingo API on first run:
- **30 companies**: Major tech (AAPL, MSFT, GOOGL, etc.) and emerging AI/energy stocks
- **60+ metrics**: Income, balance sheet, cash flow, and market metrics
- **Two data sources**: Statement data (quarterly/annual) and daily market data

### Test Server

Run the test server to verify everything works:

```bash
python server.py
```

Open http://localhost:8000 to:
- View financial data in the database
- Run model evaluations and see accuracy scores

### Training

Open `finance_rl.ipynb` in Jupyter or Google Colab to train a model with RL.

Reward signal: training uses the LLM-as-judge correctness score only.

## Files

- **`database.py`** - Database setup with Tiingo statements + daily data endpoints
- **`synthetic.py`** - Generates test cases from database
- **`environment.py`** - Financial tool execution
- **`agent.py`** - Multi-turn LLM agent
- **`rewards.py`** - LLM-as-judge evaluation
- **`rollout.py`** - Training trajectory generation
- **`server.py`** - FastAPI test server
- **`index.html`** - Web interface

## Data Coverage

The system fetches comprehensive financial data using exact Tiingo field names:
- **Income Statement**: revenue, netinc, ebitda, eps, grossProfit, etc.
- **Balance Sheet**: assets, cashAndEq, equity, debt, inventory, etc.
- **Cash Flow**: freeCashFlow, ncfo, capex, payDiv, etc.
- **Market Data**: marketCap, peRatio, pbRatio, trailingPEG1Y
- **Calculated Metrics**: roe, roa, grossMargin, currentRatio, etc.