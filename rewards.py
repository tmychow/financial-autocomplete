"""
Reward calculation module using LLM-as-judge for autocomplete evaluation
"""

import re
from typing import Tuple, Optional, Dict, Any, List
import os

async def evaluate_completion_with_judge(
    prediction: Optional[str],
    ground_truth: str,
    judge_model: str = "gpt-4.1"
) -> Tuple[bool, str, float]:
    """
    Use LLM as judge to evaluate completion quality
    This is the same approach used in the main evaluation system
    
    Args:
        prediction: Model's predicted completion
        ground_truth: Expected completion
        judge_model: Model to use as judge
        
    Returns:
        Tuple of (is_correct, reasoning, score)
        where score is 1.0 for correct, 0.0 for incorrect
    """
    from openai import AsyncOpenAI
    
    # Handle empty predictions
    if not prediction and ground_truth != "NO_COMPLETION_NEEDED":
        return False, "No prediction provided", 0.0
    
    try:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if ground_truth == "NO_COMPLETION_NEEDED":
            prompt = f"""You are evaluating whether a model correctly identified that no completion was needed.

The model's response: {prediction if prediction else "[empty/no response]"}

The model should have either:
1. Returned nothing/empty string
2. Indicated that no completion is needed

Did the model correctly avoid providing a completion?

Please respond with your judgment in the following XML format:
<judgment>
<verdict>YES/NO</verdict>
<reasoning>Your explanation for why you made this decision</reasoning>
</judgment>"""
        else:
            prompt = f"""You are evaluating whether a model's prediction matches the ground truth for a financial data AUTOCOMPLETE task.

Ground Truth (the completion only): {ground_truth}
Model Prediction: {prediction}

CRITICAL: This is an autocomplete task, so the model should return ONLY the completion text, NOT the full sentence.
For example, if the input is "Apple's revenue in 2023 was", the model should return "$383.3 billion", NOT "Apple's revenue in 2023 was $383.3 billion".

Please determine if the model's prediction is correct. Consider:
- The model MUST return only the completion suffix, not repeat the input
- Numeric values should be approximately equal (rounding is acceptable)
- Different formats are acceptable (e.g., "$1.2B" vs "$1.2 billion" vs "1200 million")
- Formatting differences are acceptable (e.g. $ vs USD, B vs billion vs billions)
- We care about the meaning, not the symbols or how natural the language is
- 0 does not mean no completion needed

Please respond with your judgment in the following XML format:
<judgment>
<verdict>YES/NO</verdict>
<reasoning>Your explanation for why you made this decision</reasoning>
</judgment>"""

        response = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse XML response
        verdict_match = re.search(r'<verdict>(YES|NO)</verdict>', response_text, re.IGNORECASE)
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL)
        
        if verdict_match:
            verdict = verdict_match.group(1).upper() == "YES"
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            score = 1.0 if verdict else 0.0
            return verdict, reasoning, score
        else:
            # Fallback if XML parsing fails
            is_correct = "YES" in response_text.upper()
            score = 1.0 if is_correct else 0.0
            return is_correct, f"Failed to parse XML response: {response_text}", score
        
    except Exception as e:
        print(f"Error using {judge_model} judge: {e}")
        # Fallback to simple exact match
        is_correct = prediction.lower().strip() == ground_truth.lower().strip() if prediction else False
        score = 1.0 if is_correct else 0.0
        return is_correct, f"Fallback evaluation due to error: {str(e)}", score

async def calculate_reward(
    prediction: Optional[str],
    ground_truth: str,
    episode_info: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    case_metadata: Optional[Dict[str, Any]] = None,
    use_judge: bool = True,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate reward for an autocomplete episode using correctness only.
    """
    # Extract episode information
    num_tool_calls = episode_info.get("tool_calls_count", 0)
    completed = episode_info.get("completed", False)
    max_turns_reached = episode_info.get("max_turns_reached", False)
    error_occurred = episode_info.get("error", False)
    
    # Calculate correctness score using judge
    if use_judge:
        # Allow explicit argument to override env variable, which overrides default
        if judge_model is None:
            judge_model = os.getenv("JUDGE_MODEL", "gpt-4.1")
        is_correct, reasoning, correctness_score = await evaluate_completion_with_judge(
            prediction, ground_truth, judge_model
        )
    else:
        # Simple string matching fallback
        is_correct = prediction == ground_truth if prediction else False
        correctness_score = 1.0 if is_correct else 0.0
        reasoning = "Simple string match evaluation"
    
    # Additional reward components
    # Weights (configurable via env)
    W_JUDGE = float(os.getenv("W_JUDGE", "1.0"))
    W_USED_SEARCH = float(os.getenv("W_USED_SEARCH", "0.1"))
    # Keep combined coverage available but default to 0 (replaced by separate components)
    W_COVERAGE = float(os.getenv("W_COVERAGE", "0.0"))
    # New separate components
    W_TICKER = float(os.getenv("W_TICKER", "0.0667"))
    W_METRIC = float(os.getenv("W_METRIC", "0.0667"))
    W_PERIOD = float(os.getenv("W_PERIOD", "0.0666"))

    # Determine if this is a no-completion case
    case_type = (case_metadata or {}).get("type") if isinstance(case_metadata, dict) else None
    is_no_completion = case_type == "no_completion" or ground_truth == "NO_COMPLETION_NEEDED"

    # Used search flag
    used_search = 0.0
    lookup_coverage = 0.0
    ticker_correct = 0.0
    metric_correct = 0.0
    period_correct = 0.0

    if tool_calls and not is_no_completion:
        used_search = 1.0 if any(tc.get("tool") == "search" for tc in tool_calls) else 0.0
        # Build observed lookups from results
        observed = set()
        for tc in tool_calls:
            if tc.get("tool") == "search":
                res = tc.get("result")
                if isinstance(res, dict):
                    mt = res.get("metric")
                    tk = res.get("ticker")
                    pr = res.get("period")
                    if mt and tk and pr:
                        observed.add((tk, mt, pr))
        # Compute coverage against required_lookups
        required = []
        if isinstance(case_metadata, dict):
            required = case_metadata.get("required_lookups") or []
        required_tuples = []
        for req in required:
            tk = req.get("ticker")
            mt = req.get("metric")
            pr = req.get("period")
            if tk and mt and pr:
                if isinstance(pr, str) and pr.lower() == "latest":
                    # If requirement is latest, consider any observed tuple with same tk/mt as match
                    # (concrete period will be in observed)
                    # Represent latest with wildcard marker
                    required_tuples.append((tk, mt, "*latest*"))
                else:
                    required_tuples.append((tk, mt, pr))
        matched = 0
        total = len(required_tuples)
        if total > 0:
            for r in required_tuples:
                if r[2] == "*latest*":
                    if any((tk, mt) == (r[0], r[1]) for tk, mt, _ in observed):
                        matched += 1
                else:
                    if (r[0], r[1], r[2]) in observed:
                        matched += 1
            lookup_coverage = matched / total if total else 0.0

            # Separate components: ticker, metric, period correctness
            # ticker_correct: for each required, did any observed call contain the required ticker (regardless of metric/period)?
            ticker_hits = 0
            metric_hits = 0
            period_hits = 0
            observed_tickers = {tk for (tk, _, __) in observed}
            observed_metrics = {mt for (_, mt, __) in observed}
            for tk, mt, pr in required_tuples:
                if tk in observed_tickers:
                    ticker_hits += 1
                if mt in observed_metrics:
                    metric_hits += 1
                # Period check: if "latest", accept any observed with same (tk, mt)
                if pr == "*latest*":
                    if any((otk, omt) == (tk, mt) for (otk, omt, opr) in observed):
                        period_hits += 1
                else:
                    if any(opr == pr for (_, _, opr) in observed):
                        period_hits += 1
            ticker_correct = ticker_hits / total
            metric_correct = metric_hits / total
            period_correct = period_hits / total

    # Total reward
    total_reward = (
        W_JUDGE * correctness_score
        + (0.0 if is_no_completion else W_USED_SEARCH * used_search)
        + (0.0 if is_no_completion else W_COVERAGE * lookup_coverage)
        + (0.0 if is_no_completion else W_TICKER * ticker_correct)
        + (0.0 if is_no_completion else W_METRIC * metric_correct)
        + (0.0 if is_no_completion else W_PERIOD * period_correct)
    )

    return {
        "total_reward": total_reward,
        "correctness_score": correctness_score,
        "is_correct": is_correct,
        "num_tool_calls": num_tool_calls,
        "completed": completed,
        "max_turns_reached": max_turns_reached,
        "error_occurred": error_occurred,
        "reasoning": reasoning,
        "used_search": used_search,
        "lookup_coverage": lookup_coverage,
        "ticker_correct": ticker_correct,
        "metric_correct": metric_correct,
        "period_correct": period_correct,
    }

# ============== Batch Reward Calculation ==============

async def calculate_batch_rewards(
    predictions: List[Optional[str]],
    ground_truths: List[str],
    episode_infos: List[Dict[str, Any]],
    tool_calls_list: Optional[List[List[Dict[str, Any]]]] = None,
    case_metadatas: Optional[List[Dict[str, Any]]] = None,
    use_judge: bool = True,
    judge_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Calculate rewards for a batch of episodes
    
    Args:
        predictions: List of model predictions
        ground_truths: List of expected completions
        episode_infos: List of episode information dictionaries
        use_judge: Whether to use LLM judge
        
    Returns:
        List of reward dictionaries
    """
    rewards = []
    
    tool_calls_list = tool_calls_list or [None] * len(predictions)
    case_metadatas = case_metadatas or [None] * len(predictions)

    for pred, truth, info, tcs, meta in zip(predictions, ground_truths, episode_infos, tool_calls_list, case_metadatas):
        reward = await calculate_reward(
            pred, truth, info, tool_calls=tcs, case_metadata=meta, use_judge=use_judge, judge_model=judge_model
        )
        rewards.append(reward)
    
    return rewards

# ============== Testing ==============

if __name__ == "__main__":
    import asyncio
    
    async def test_rewards():
        # Test cases
        test_cases = [
            {
                "prediction": "$383.3 billion",
                "ground_truth": "$383.3 billion",
                "episode_info": {
                    "tool_calls_count": 4,
                    "completed": True,
                    "max_turns_reached": False
                }
            },
            {
                "prediction": "$383 billion",  # Close but not exact
                "ground_truth": "$383.3 billion",
                "episode_info": {
                    "tool_calls_count": 5,
                    "completed": True,
                    "max_turns_reached": False
                }
            },
            {
                "prediction": "NO_COMPLETION_NEEDED",
                "ground_truth": "NO_COMPLETION_NEEDED",
                "episode_info": {
                    "tool_calls_count": 1,
                    "completed": True,
                    "max_turns_reached": False
                }
            },
            {
                "prediction": None,  # Failed to complete
                "ground_truth": "$100 billion",
                "episode_info": {
                    "tool_calls_count": 10,
                    "completed": False,
                    "max_turns_reached": True
                }
            }
        ]
        
        print("Testing reward calculation...")
        
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest case {i}:")
            print(f"  Prediction: {case['prediction']}")
            print(f"  Ground truth: {case['ground_truth']}")
            print(f"  Episode: {case['episode_info']}")
            
            # Test without judge (simple matching)
            reward = await calculate_reward(
                case["prediction"],
                case["ground_truth"],
                case["episode_info"],
                use_judge=False
            )
            
            print(f"  Reward (no judge): {reward['total_reward']:.3f}")
            print(f"    - Correctness: {reward['correctness_score']:.3f}")
            
            # Uncomment to test with judge (requires OpenAI API key)
            # reward_with_judge = await calculate_reward(
            #     case["prediction"],
            #     case["ground_truth"],
            #     case["episode_info"],
            #     use_judge=True
            # )
            # print(f"  Reward (with judge): {reward_with_judge['total_reward']:.3f}")
            # print(f"    Reasoning: {reward_with_judge['reasoning']}")
    
    asyncio.run(test_rewards())