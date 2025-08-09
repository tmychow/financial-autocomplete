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
- Numeric values should be approximately equal (rounding to low precision is acceptable)
- Different formats are acceptable (e.g., "$1.2B" vs "$1.2 billion" vs "1200 million")
- Formatting differences (e.g. $ vs USD, B vs billion) are acceptable
- We care about the meaning, not the exact symbols or how natural the language is
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

def calculate_efficiency_bonus(
    num_tool_calls: int,
    completed: bool,
    max_expected_calls: int = 6
) -> float:
    """
    Calculate efficiency bonus based on number of tool calls
    
    Args:
        num_tool_calls: Number of tool calls made
        completed: Whether the episode completed successfully
        max_expected_calls: Maximum expected number of calls for simple tasks
        
    Returns:
        Efficiency bonus (0.0 to 0.2)
    """
    if not completed:
        return 0.0
    
    # Fewer calls is better
    if num_tool_calls <= max_expected_calls:
        # Linear bonus from 0.2 (at 1 call) to 0.0 (at max_expected_calls)
        return max(0.0, 0.2 * (1 - (num_tool_calls - 1) / max_expected_calls))
    else:
        # Penalty for too many calls
        return max(-0.2, -0.05 * (num_tool_calls - max_expected_calls))

def calculate_completion_penalty(
    completed: bool,
    max_turns_reached: bool,
    error_occurred: bool = False
) -> float:
    """
    Calculate penalty for incomplete or error episodes
    
    Args:
        completed: Whether episode completed with return_answer
        max_turns_reached: Whether max turns was reached
        error_occurred: Whether an error occurred
        
    Returns:
        Penalty value (negative)
    """
    if error_occurred:
        return -1.0  # Severe penalty for errors
    
    if not completed:
        if max_turns_reached:
            return -0.5  # Moderate penalty for timeout
        else:
            return -0.3  # Smaller penalty for other incompletions
    
    return 0.0

async def calculate_reward(
    prediction: Optional[str],
    ground_truth: str,
    episode_info: Dict[str, Any],
    use_judge: bool = True,
    efficiency_weight: float = 0.2,
    completion_weight: float = 0.3
) -> Dict[str, Any]:
    """
    Calculate comprehensive reward for an autocomplete episode
    
    Args:
        prediction: Model's predicted completion
        ground_truth: Expected completion
        episode_info: Information about the episode (turns, tool_calls, etc.)
        use_judge: Whether to use LLM judge (vs simple match)
        efficiency_weight: Weight for efficiency bonus
        completion_weight: Weight for completion penalty
        
    Returns:
        Dictionary with reward components and total reward
    """
    # Extract episode information
    num_tool_calls = episode_info.get("tool_calls_count", 0)
    completed = episode_info.get("completed", False)
    max_turns_reached = episode_info.get("max_turns_reached", False)
    error_occurred = episode_info.get("error", False)
    
    # Calculate correctness score using judge
    if use_judge:
        is_correct, reasoning, correctness_score = await evaluate_completion_with_judge(
            prediction, ground_truth
        )
    else:
        # Simple string matching fallback
        is_correct = prediction == ground_truth if prediction else False
        correctness_score = 1.0 if is_correct else 0.0
        reasoning = "Simple string match evaluation"
    
    # Calculate efficiency bonus
    efficiency_bonus = calculate_efficiency_bonus(num_tool_calls, completed)
    
    # Calculate completion penalty
    completion_penalty = calculate_completion_penalty(
        completed, max_turns_reached, error_occurred
    )
    
    # Use only LLM judge score as the reward
    total_reward = correctness_score
    
    return {
        "total_reward": total_reward,
        "correctness_score": correctness_score,
        "is_correct": is_correct,
        "efficiency_bonus": efficiency_bonus,
        "completion_penalty": completion_penalty,
        "num_tool_calls": num_tool_calls,
        "completed": completed,
        "reasoning": reasoning,
        "components": {
            "base": correctness_score,
            "efficiency": efficiency_weight * efficiency_bonus,
            "completion": completion_weight * completion_penalty
        }
    }

# ============== Batch Reward Calculation ==============

async def calculate_batch_rewards(
    predictions: List[Optional[str]],
    ground_truths: List[str],
    episode_infos: List[Dict[str, Any]],
    use_judge: bool = True
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
    
    for pred, truth, info in zip(predictions, ground_truths, episode_infos):
        reward = await calculate_reward(pred, truth, info, use_judge)
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
            print(f"    - Efficiency: {reward['efficiency_bonus']:.3f}")
            print(f"    - Completion: {reward['completion_penalty']:.3f}")
            
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