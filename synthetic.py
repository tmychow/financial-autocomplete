"""
Synthetic data generation for Finance RL Autocomplete Training
Generates test cases from the financial database
"""

import asyncio
import random
from typing import List, Dict, Optional
from database import (
    get_all_metrics, get_available_periods, get_financial_value,
    get_db, get_tickers_with_data, get_latest_period
)

# ============== Formatting Helpers ==============

def format_value(value: float, unit: str) -> str:
    """Format a financial value with appropriate unit - using full precision to match what model sees"""
    if unit == "USD_billions":
        return f"${value} billion"  # No rounding - use exact value
    if unit == "USD":
        return f"${value}"  # No rounding
    if unit == "percentage":
        # Value is already stored as percentage (e.g., 14.5 for 14.5%)
        return f"{value}%"  # No rounding
    if unit == "count":
        return f"{int(round(value)):,}"  # Keep rounding for counts since they should be integers
    if unit == "ratio":
        return str(value)  # No rounding
    return str(value)

async def get_company_name(ticker: str) -> str:
    """Get company name for a ticker"""
    async with get_db() as db:
        async with db.execute(
            "SELECT company_name FROM tickers WHERE ticker = ?",
            (ticker,)
        ) as cur:
            row = await cur.fetchone()
            return row["company_name"] if row and row["company_name"] else ticker

# ============== Generation Helpers ==============

async def random_period(ticker: str, metric: str) -> Optional[str]:
    """Get a random period for ticker/metric combination"""
    periods = await get_available_periods(ticker, metric)
    return random.choice(periods) if periods else None

# ============== Test Case Generators ==============

async def generate_latest_case(tickers: List[str], metrics: List[Dict[str, str]]):
    """Generate a case explicitly using 'latest' keyword"""
    for _ in range(10):
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        
        # Get the latest period for this ticker/metric
        latest_period = await get_latest_period(ticker, metric["metric_name"])
        if not latest_period:
            continue
        
        value = await get_financial_value(ticker, metric["metric_name"], latest_period)
        if not value:
            continue
        
        company = await get_company_name(ticker)
        
        # Templates that explicitly use "latest"
        templates = [
            "{company}'s latest {desc} is ",
            "The latest {desc} for {company} is ",
            "{company} latest reported {desc} of ",
            "Latest {desc} for {company}: ",
            "{company}'s most recent {desc} is ",
        ]
        
        prefix = random.choice(templates).format(
            company=company,
            desc=metric["description"].lower()
        )
        completion = format_value(value["value"], value.get("unit"))
        return {"input": prefix, "ground_truth": completion}
    return None

async def generate_simple_case(tickers: List[str], metrics: List[Dict[str, str]]):
    """Generate a simple value lookup case"""
    for _ in range(10):
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        
        # Templates with explicit period placeholders (use random period)
        templates_with_period = [
            "{company}'s {desc} in {period} was ",
            "The {desc} for {company} in {period} was ",
            "In {period}, {company}'s {desc} was ",
            "For {period}, {company} had {desc} of ",
        ]
        
        # Templates without period (should use latest period)
        templates_without_period = [
            "{company} reported {desc} of ",
            "{company}'s {desc} is ",
            "The {desc} for {company} is ",
            "{company} has {desc} of ",
        ]
        
        # Randomly choose which type of template to use
        use_period_template = random.random() < 0.6  # 60% with period, 40% without
        
        if use_period_template:
            period = await random_period(ticker, metric["metric_name"])
            if not period:
                continue
            template = random.choice(templates_with_period)
            prefix = template.format(
                company=await get_company_name(ticker),
                desc=metric["description"].lower(),
                period=period,
            )
        else:
            # Use latest period for templates without period
            period = await get_latest_period(ticker, metric["metric_name"])
            if not period:
                continue
            template = random.choice(templates_without_period)
            prefix = template.format(
                company=await get_company_name(ticker),
                desc=metric["description"].lower(),
            )
        
        value = await get_financial_value(ticker, metric["metric_name"], period)
        if not value:
            continue
        
        completion = format_value(value["value"], value.get("unit"))
        return {"input": prefix, "ground_truth": completion}
    return None

async def generate_difference_case(tickers: List[str], metrics: List[Dict[str, str]]):
    """Generate a case comparing values across periods"""
    for _ in range(10):
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        periods = await get_available_periods(ticker, metric["metric_name"])
        if len(periods) < 2:
            continue
        p1, p2 = sorted(random.sample(periods, 2))
        v1 = await get_financial_value(ticker, metric["metric_name"], p1)
        v2 = await get_financial_value(ticker, metric["metric_name"], p2)
        if not v1 or not v2:
            continue
        diff = v2["value"] - v1["value"]
        company = await get_company_name(ticker)
        
        templates = [
            "The change in {desc} for {company} from {p1} to {p2} is ",
            "{company}'s {desc} changed from {p1} to {p2} by ",
            "From {p1} to {p2}, {company}'s {desc} difference was ",
        ]
        
        prefix = random.choice(templates).format(
            company=company,
            desc=metric['description'].lower(),
            p1=p1,
            p2=p2
        )
        completion = format_value(diff, v1.get("unit"))
        return {"input": prefix, "ground_truth": completion}
    return None

async def generate_cagr_case(tickers: List[str], metrics: List[Dict[str, str]]):
    """Generate CAGR calculation case"""
    for _ in range(10):
        ticker = random.choice(tickers)
        metric = random.choice(metrics)
        # Only use FY periods for CAGR
        periods = [p for p in await get_available_periods(ticker, metric["metric_name"]) if p.endswith("FY")]
        if len(periods) < 2:
            continue
        start, end = sorted(random.sample(periods, 2))
        v_start = await get_financial_value(ticker, metric["metric_name"], start)
        v_end = await get_financial_value(ticker, metric["metric_name"], end)
        if not v_start or not v_end:
            continue
        years = int(end[:4]) - int(start[:4])
        if years <= 0 or v_start["value"] == 0 or v_end["value"] == 0:
            continue
        
        # Handle negative values for CAGR
        if (v_start["value"] < 0 and v_end["value"] > 0) or (v_start["value"] > 0 and v_end["value"] < 0):
            continue  # Skip sign changes
        
        if v_start["value"] < 0 and v_end["value"] < 0:
            # Both negative - use absolute values
            cagr = ((abs(v_end["value"]) / abs(v_start["value"])) ** (1 / years) - 1) * 100
        else:
            # Both positive
            cagr = ((v_end["value"] / v_start["value"]) ** (1 / years) - 1) * 100
        
        company = await get_company_name(ticker)
        prefix = f"The CAGR of {metric['description'].lower()} for {company} from {start} to {end} is "
        completion = f"{cagr:.1f}%"
        return {"input": prefix, "ground_truth": completion}
    return None

async def generate_cross_ticker_difference_case(tickers: List[str], metrics: List[Dict[str, str]]):
    """Generate case comparing two companies"""
    for _ in range(10):
        if len(tickers) < 2:
            continue
        metric = random.choice(metrics)
        t1, t2 = random.sample(tickers, 2)
        
        # Find periods where BOTH tickers have data for this metric
        periods_with_both = []
        periods1 = await get_available_periods(t1, metric["metric_name"])
        
        for period in periods1:
            v1 = await get_financial_value(t1, metric["metric_name"], period)
            v2 = await get_financial_value(t2, metric["metric_name"], period)
            if v1 and v2:
                periods_with_both.append((period, v1, v2))
        
        if not periods_with_both:
            continue
        
        company1 = await get_company_name(t1)
        company2 = await get_company_name(t2)
        
        # Templates with period (use random)
        templates_with_period = [
            "The difference in {desc} between {c1} and {c2} in {period} was ",
            "In {period}, the {desc} gap between {c1} and {c2} was ",
            "{c1} vs {c2} {desc} difference in {period}: ",
        ]
        
        # Templates without period (use latest matching)
        templates_without_period = [
            "The difference in {desc} between {c1} and {c2} is ",
            "The {desc} gap between {c1} and {c2} is ",
            "{c1} vs {c2} {desc} difference: ",
        ]
        
        use_period_template = random.random() < 0.6
        
        if use_period_template:
            period, v1, v2 = random.choice(periods_with_both)
            template = random.choice(templates_with_period)
            prefix = template.format(
                desc=metric['description'].lower(),
                c1=company1,
                c2=company2,
                period=period
            )
        else:
            # Use the latest period (first in the sorted list)
            period, v1, v2 = periods_with_both[0]  # Already sorted DESC from database
            template = random.choice(templates_without_period)
            prefix = template.format(
                desc=metric['description'].lower(),
                c1=company1,
                c2=company2
            )
        
        # Don't use abs() - keep the actual difference
        diff = v1["value"] - v2["value"]
        completion = format_value(diff, v1.get("unit"))
        return {"input": prefix, "ground_truth": completion}
    return None

# Predefined calculation combinations
CALC_COMBOS = [
    {
        "m1": "netinc", "m2": "revenue",
        "operation": "divide", "unit": "percentage",
        "description": "net profit margin"
    },
    {
        "m1": "grossProfit", "m2": "revenue",
        "operation": "divide", "unit": "percentage",
        "description": "gross margin"
    },
    {
        "m1": "debt", "m2": "equity",
        "operation": "divide", "unit": "ratio",
        "description": "debt to equity ratio"
    },
    {
        "m1": "revenue", "m2": "costRev",
        "operation": "subtract", "unit": "USD_billions",
        "description": "gross profit (calculated)"
    },
    {
        "m1": "freeCashFlow", "m2": "capex",
        "operation": "add", "unit": "USD_billions",
        "description": "operating cash flow approximation"
    },
]

async def generate_multi_metric_calc_case(tickers: List[str], metrics: List[Dict[str, str]]):
    """Generate case requiring calculation between metrics"""
    for _ in range(10):
        ticker = random.choice(tickers)
        combo = random.choice(CALC_COMBOS)
        
        # Get periods where BOTH metrics have data
        periods_with_both = []
        periods1 = await get_available_periods(ticker, combo["m1"])
        
        for period in periods1:
            v1 = await get_financial_value(ticker, combo["m1"], period)
            v2 = await get_financial_value(ticker, combo["m2"], period)
            if v1 and v2:
                periods_with_both.append((period, v1, v2))
        
        if not periods_with_both:
            continue
        
        company = await get_company_name(ticker)
        
        # Decide whether to use specific period or latest
        use_period_template = random.random() < 0.5
        
        if use_period_template:
            period, v1, v2 = random.choice(periods_with_both)
            prefix = f"The {combo['description']} for {company} in {period} was "
        else:
            # Use latest period (first in sorted list)
            period, v1, v2 = periods_with_both[0]
            prefix = f"The {combo['description']} for {company} is "
        
        if combo["operation"] == "divide":
            if v2["value"] == 0:
                continue
            result = v1["value"] / v2["value"]
            if combo["unit"] == "percentage":
                # Check if values are already in percentage form
                # If both values are in billions/regular numbers, convert to percentage
                if v1.get("unit") != "percentage" and v2.get("unit") != "percentage":
                    result *= 100
        elif combo["operation"] == "subtract":
            result = v1["value"] - v2["value"]
        elif combo["operation"] == "add":
            result = v1["value"] + v2["value"]
        else:
            continue
        
        completion = format_value(result, combo["unit"])
        return {"input": prefix, "ground_truth": completion}
    return None

# No-completion prefixes
STATIC_NO_COMPLETION_PREFIXES = [
    "The CFO mentioned during the call that ",
    "This quarter the company expects that ",
    "Financial analysts often say that ",
    "The board announced today that ",
    "According to the press release, ",
    "During the investor day they said that ",
    "The CEO remarked in the interview that ",
    "Analysts on Wall Street are predicting ",
    "Management highlighted in the annual report that ",
    "In recent news articles it was reported that ",
    "The company's guidance suggests that ",
    "Market sentiment indicates that ",
]

DYNAMIC_NO_COMPLETION_TEMPLATES = [
    "{company}'s {desc} can be broken down into ",
    "Analysts often track {company}'s {desc} closely because ",
    "During the {period} call, management discussed {company}'s {desc} and ",
    "{company}'s team mentioned its {desc} trend during the ",
    "There has been speculation about how {company}'s {desc} might ",
    "The market reaction to {company}'s {desc} was ",
]

async def generate_no_completion_case(tickers: List[str], metrics: List[Dict[str, str]]) -> Dict[str, str]:
    """Generate case where no completion is needed"""
    if random.random() < 0.5:
        prefix = random.choice(STATIC_NO_COMPLETION_PREFIXES)
    else:
        ticker = random.choice(tickers) if tickers else "AAPL"
        metric = random.choice(metrics) if metrics else {"description": "revenue"}
        period = await random_period(ticker, metric["metric_name"]) if metrics else "2023Q4"
        company = await get_company_name(ticker)
        template = random.choice(DYNAMIC_NO_COMPLETION_TEMPLATES)
        prefix = template.format(
            company=company,
            desc=metric["description"].lower(),
            period=period or "the last quarter"
        )
    return {"input": prefix, "ground_truth": "NO_COMPLETION_NEEDED"}

# ============== Main Generation Function ==============

async def generate_cases(num_cases: int, no_completion_ratio: float = 0.25) -> List[Dict[str, str]]:
    """
    Generate synthetic test cases
    
    Args:
        num_cases: Number of cases to generate
        no_completion_ratio: Ratio of no-completion cases (default 0.25)
    """
    tickers = await get_tickers_with_data()
    metrics = await get_all_metrics()
    
    if not tickers or not metrics:
        print("Warning: No tickers or metrics found in database")
        return []
    
    cases = []
    
    # Generators with their weights
    generators = [
        generate_simple_case,           # 35%
        generate_latest_case,           # 10%
        generate_difference_case,        # 15%
        generate_cross_ticker_difference_case,  # 10%
        generate_multi_metric_calc_case,       # 15%
        generate_cagr_case,             # 15%
    ]
    weights = [0.35, 0.1, 0.15, 0.1, 0.15, 0.15]
    
    attempts = 0
    max_attempts = num_cases * 10  # Prevent infinite loop
    
    while len(cases) < num_cases and attempts < max_attempts:
        attempts += 1
        
        # Decide if this should be a no-completion case
        if random.random() < no_completion_ratio:
            case = await generate_no_completion_case(tickers, metrics)
            cases.append(case)
            continue
        
        # Choose a generator based on weights
        gen = random.choices(generators, weights)[0]
        case = await gen(tickers, metrics)
        if case:
            cases.append(case)
    
    if len(cases) < num_cases:
        print(f"Warning: Only generated {len(cases)} cases out of {num_cases} requested")
    
    return cases

# ============== Batch Generation for Training ==============

async def generate_training_data(
    num_train: int = 200,
    num_eval: int = 50,
    num_sample: int = 10
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate train, eval, and sample datasets
    
    Returns:
        Dictionary with 'train', 'eval', and 'sample' keys
    """
    print(f"Generating {num_train} training cases...")
    train_cases = await generate_cases(num_train)
    
    print(f"Generating {num_eval} evaluation cases...")
    eval_cases = await generate_cases(num_eval)
    
    print(f"Generating {num_sample} sample cases...")
    sample_cases = await generate_cases(num_sample)
    
    return {
        "train": train_cases,
        "eval": eval_cases,
        "sample": sample_cases
    }

if __name__ == "__main__":
    # Test synthetic data generation
    async def test():
        from database import setup_database
        
        # Setup database with sample data
        await setup_database()  # Requires TIINGO_API_KEY
        
        # Generate some test cases
        cases = await generate_cases(10)
        print(f"\nGenerated {len(cases)} test cases:")
        for i, case in enumerate(cases, 1):
            print(f"\n{i}. Input: {case['input']}")
            print(f"   Expected: {case['ground_truth']}")
    
    asyncio.run(test())