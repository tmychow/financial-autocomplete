"""
Synthetic data generation for Finance RL Autocomplete Training
Generates test cases from the financial database
"""

import asyncio
import random
import os
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
    """Get colloquial company name for a ticker (prefer short/common names)."""
    async with get_db() as db:
        async with db.execute(
            "SELECT company_name FROM tickers WHERE ticker = ?",
            (ticker,)
        ) as cur:
            row = await cur.fetchone()
            full_name = row["company_name"] if row and row["company_name"] else ticker
            # Comprehensive map for all loaded companies (see database.setup_database tickers_to_load)
            mapping = {
                # Tech megacaps
                "Apple Inc.": "Apple",
                "Microsoft Corporation": "Microsoft",
                "Alphabet Inc.": "Google",
                "Amazon.com Inc.": "Amazon",
                "Meta Platforms Inc.": "Meta",
                # Semis / chip-related
                "NVIDIA Corporation": "NVIDIA",
                "Intel Corporation": "Intel",
                "Micron Technology Inc.": "Micron",
                "Taiwan Semiconductor Manufacturing": "TSMC",
                "Broadcom Inc.": "Broadcom",
                "Marvell Technology Inc.": "Marvell",
                "Applied Materials Inc.": "Applied Materials",
                "Lam Research Corporation": "Lam Research",
                "ASML Holding N.V.": "ASML",
                # Software / enterprise
                "Oracle Corporation": "Oracle",
                "Arista Networks Inc.": "Arista Networks",
                # Energy / utilities / industrials
                "Tesla Inc.": "Tesla",
                "Vistra Corp.": "Vistra",
                "GE Vernova Inc.": "GE Vernova",
                "Constellation Energy Corporation": "Constellation Energy",
                "NRG Energy Inc.": "NRG Energy",
                "Talen Energy Corporation": "Talen Energy",
                "Bloom Energy Corporation": "Bloom Energy",
                # Edge / communications / components
                "Coherent Corp.": "Coherent",
                "Onto Innovation Inc.": "Onto Innovation",
                "Credo Technology Group Holding Ltd": "Credo Technology",
                "Fabrinet": "Fabrinet",
                "TSS Inc.": "TSS",
                # Data center / infrastructure
                "Vertiv Holdings Co.": "Vertiv",
                # Newer listings / others
                "Nebius Group N.V.": "Nebius",
            }
            if full_name in mapping:
                return mapping[full_name]

            # Generic cleanup fallback: remove common corporate suffixes
            cleaned = full_name
            # Remove common suffixes once, insensitive to dots/commas
            suffixes = [
                " Inc.", " Incorporated", " Corporation", " Corp.", " Co.",
                " N.V.", " Ltd", " Limited", " Holdings", " Holding", " Group"
            ]
            for suf in suffixes:
                if cleaned.endswith(suf):
                    cleaned = cleaned[: -len(suf)]
                    cleaned = cleaned.rstrip()
            return cleaned

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
        
        # Templates that explicitly use "latest" (expanded for diversity)
        templates = [
            "{company}'s latest {desc} is ",
            "The latest {desc} for {company} is ",
            "{company} latest reported {desc} of ",
            "Latest {desc} for {company}: ",
            "{company}'s most recent {desc} is ",
            "As of the latest period, {company}'s {desc} is ",
            "Most recent {desc} reported by {company} is ",
            "Latest available {desc} for {company} stands at ",
            "For the latest period, {company}'s {desc} was ",
            "In the most recent filing, {company}'s {desc} was ",
        ]
        
        prefix = random.choice(templates).format(
            company=company,
            desc=metric["description"].lower()
        )
        completion = format_value(value["value"], value.get("unit"))
        metadata = {
            "type": "latest",
            "required_lookups": [
                {"ticker": ticker, "metric": metric["metric_name"], "period": "latest"}
            ]
        }
        return {"input": prefix, "ground_truth": completion, "metadata": metadata}
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
            "During {period}, {company} reported {desc} was ",
            "Across {period}, {company}'s {desc} came in at ",
            "For the period {period}, {company}'s {desc} was ",
            "{period} saw {company}'s {desc} at ",
        ]
        
        # # Templates without period (should use latest period)
        # templates_without_period = [
        #     "{company} reported {desc} of ",
        #     "{company}'s {desc} is ",
        #     "The {desc} for {company} is ",
        #     "{company} has {desc} of ",
        #     "Currently, {company}'s {desc} is ",
        #     "For {company}, {desc} is ",
        #     "{company} shows {desc} of ",
        # ]
        
        # Randomly choose which type of template to use
        use_period_template = random.random() < 1.1  # 100% with period
        
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
        # else:
        #     # Use latest period for templates without period
        #     period = await get_latest_period(ticker, metric["metric_name"])
        #     if not period:
        #         continue
        #     template = random.choice(templates_without_period)
        #     prefix = template.format(
        #         company=await get_company_name(ticker),
        #         desc=metric["description"].lower(),
        #     )
        
        value = await get_financial_value(ticker, metric["metric_name"], period)
        if not value:
            continue
        
        completion = format_value(value["value"], value.get("unit"))
        metadata = {
            "type": "simple",
            "required_lookups": [
                {"ticker": ticker, "metric": metric["metric_name"], "period": period}
            ]
        }
        return {"input": prefix, "ground_truth": completion, "metadata": metadata}
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
            "From {p1} to {p2}, {company}'s {desc} changed by ",
            "Between {p1} and {p2}, {company}'s {desc} moved by ",
            "Change in {company}'s {desc} from {p1} to {p2}: ",
            "{company} saw its {desc} shift from {p1} to {p2} by ",
        ]
        
        prefix = random.choice(templates).format(
            company=company,
            desc=metric['description'].lower(),
            p1=p1,
            p2=p2
        )
        completion = format_value(diff, v1.get("unit"))
        metadata = {
            "type": "difference",
            "required_lookups": [
                {"ticker": ticker, "metric": metric["metric_name"], "period": p1},
                {"ticker": ticker, "metric": metric["metric_name"], "period": p2},
            ]
        }
        return {"input": prefix, "ground_truth": completion, "metadata": metadata}
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
        ]
        
        # Templates without period (use latest matching)
        templates_without_period = [
            "The difference in {desc} between {c1} and {c2} is ",
            "The {desc} gap between {c1} and {c2} is ",
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
        metadata = {
            "type": "cross_ticker_difference",
            "required_lookups": [
                {"ticker": t1, "metric": metric["metric_name"], "period": period},
                {"ticker": t2, "metric": metric["metric_name"], "period": period},
            ]
        }
        return {"input": prefix, "ground_truth": completion, "metadata": metadata}
    return None

# Predefined calculation combinations (avoid duplicates of raw metrics)
CALC_COMBOS = [
    {
        "m1": "opinc", "m2": "revenue",
        "operation": "divide", "unit": "percentage",
        "description": "operating margin"
    },
    {
        "m1": "ebitda", "m2": "revenue",
        "operation": "divide", "unit": "percentage",
        "description": "EBITDA margin"
    },
    {
        "m1": "freeCashFlow", "m2": "revenue",
        "operation": "divide", "unit": "percentage",
        "description": "free cash flow margin"
    },
    {
        "m1": "capex", "m2": "revenue",
        "operation": "divide", "unit": "percentage",
        "description": "capex to revenue"
    },
    {
        "m1": "debt", "m2": "ebitda",
        "operation": "divide", "unit": "ratio",
        "description": "debt to EBITDA"
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
        metadata = {
            "type": "calc",
            "required_lookups": [
                {"ticker": ticker, "metric": combo["m1"], "period": period},
                {"ticker": ticker, "metric": combo["m2"], "period": period},
            ],
            "calc": {"operation": combo["operation"]}
        }
        return {"input": prefix, "ground_truth": completion, "metadata": metadata}
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
    "On the earnings call, it was noted that ",
    "Investor presentations have stated that ",
    "Industry chatter points to the idea that ",
]

DYNAMIC_NO_COMPLETION_TEMPLATES = [
    "{company}'s {desc} can be broken down into ",
    "Analysts often track {company}'s {desc} closely because ",
    "During the {period} call, management discussed {company}'s {desc} and ",
    "{company}'s team mentioned its {desc} trend during the ",
    "There has been speculation about how {company}'s {desc} might ",
    "The market reaction to {company}'s {desc} was ",
    "Commentary around {company}'s {desc} has focused on ",
    "Some observers note that {company}'s {desc} could ",
    "Debate continues on whether {company}'s {desc} will ",
    "Looking back at {period}, discussion of {company}'s {desc} centered on ",
]

async def generate_no_completion_case(tickers: List[str], metrics: List[Dict[str, str]]) -> Dict[str, str]:
    """Generate case where no completion is needed"""
    if random.random() < 0.1:
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
    metadata = {"type": "no_completion", "required_lookups": []}
    return {"input": prefix, "ground_truth": "NO_COMPLETION_NEEDED", "metadata": metadata}

# ============== Main Generation Function ==============

async def generate_cases(
    num_cases: int,
    no_completion_ratio: float = 0.10,
    curriculum_stage: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Generate synthetic test cases
    
    Args:
        num_cases: Number of cases to generate
        no_completion_ratio: Ratio of no-completion cases (default 0.1)
    """
    tickers = await get_tickers_with_data()
    metrics = await get_all_metrics()
    
    if not tickers or not metrics:
        print("Warning: No tickers or metrics found in database")
        return []
    
    cases = []

    # Generators in fixed order
    generators = [
        generate_simple_case,
        generate_latest_case,
        generate_difference_case,
        generate_cross_ticker_difference_case,
        generate_multi_metric_calc_case,
    ]

    # Default distribution within completion cases
    default_weights = [0.40, 0.15, 0.15, 0.15, 0.15]

    # Curriculum schedule (stage-specific weights and no-completion ratios)
    # Stage 1: emphasize no-completion + simple/latest
    # Stage 2: introduce differences
    # Stage 3+: full mix
    stage_to_weights = {
        0: ([0.70, 0.30, 0.00, 0.00, 0.00], 0.00),
        1: ([0.70, 0.30, 0.00, 0.00, 0.00], 0.10),
        2: ([0.50, 0.20, 0.15, 0.15, 0.00], 0.10),
        3: ([0.40, 0.15, 0.15, 0.15, 0.15], 0.10),
    }

    # Select weights and no-completion ratio based on curriculum stage (if provided)
    if curriculum_stage is not None:
        weights, stage_no_completion_ratio = stage_to_weights.get(
            int(curriculum_stage), (default_weights, no_completion_ratio)
        )
        no_completion_ratio = stage_no_completion_ratio
    else:
        weights = default_weights
    
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
        # If all weights are zero (shouldn't happen), fall back to defaults
        if sum(weights) <= 0:
            weights = default_weights
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