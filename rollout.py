"""
Rollout module for generating autocomplete trajectories
Handles episode generation and trajectory building for ART training
"""

import asyncio
import time
import uuid
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio

# ART will be imported in the notebook
try:
    import art
except ImportError:
    # Mock for testing without ART
    class MockTrajectory:
        def __init__(self, messages_and_choices, reward, metrics):
            self.messages_and_choices = messages_and_choices
            self.reward = reward
            self.metrics = metrics
    
    class MockTrajectoryGroup:
        def __init__(self, trajectories):
            self.trajectories = trajectories
    
    class art:
        Trajectory = MockTrajectory
        TrajectoryGroup = MockTrajectoryGroup

from agent import AutocompleteAgent
from rewards import calculate_reward
from synthetic import generate_cases
from database import get_tickers_with_data, get_all_metrics, get_all_periods


# ============== Logging Helpers ==============

def _append_jsonl(path: str, obj: dict) -> None:
    """Append a JSON object as one line to a .jsonl file, creating directories as needed."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        # Best-effort logging; do not crash rollouts due to logging errors
        print(f"Warning: failed to write log to {path}: {e}")


def _make_rollout_log_record(
    *,
    timestamp: str,
    step: int,
    rollout_id: int,
    test_case: Dict[str, Any],
    completion: Any,
    conversation: list,
    tool_calls: list,
    episode_info: dict,
    reward_info: dict,
    latency: float,
    model: Any,
    judge_model: Optional[str],
    is_validation: bool = False,
    benchmark_reward: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a stable, human-readable log record for a rollout."""
    model_name = getattr(model, "name", None) or getattr(model, "inference_model_name", None) or "unknown"
    return {
        "timestamp": timestamp,
        "step": step,
        "rollout_id": rollout_id,
        "is_validation": is_validation,
        "input": test_case.get("input"),
        "ground_truth": test_case.get("ground_truth"),
        "prediction": completion,
        "conversation": conversation,
        "tool_calls": tool_calls,
        "episode_info": episode_info,
        "reward_info": reward_info,
        "latency_sec": latency,
        "model": model_name,
        "judge_model": judge_model or os.getenv("JUDGE_MODEL", "gpt-4.1"),
        "benchmark_reward": benchmark_reward,
    }

async def run_single_rollout(
    model: Any,
    test_case: Dict[str, str],
    rollout_id: int,
    step: int,
    use_judge: bool = True,
    max_turns: int = 7,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single autocomplete rollout
    
    Args:
        model: ART model to use
        test_case: Dictionary with 'input' and 'ground_truth'
        rollout_id: ID for this rollout
        step: Training step number
        use_judge: Whether to use LLM judge for rewards
        max_turns: Maximum conversation turns
        
    Returns:
        Dictionary containing trajectory and metadata
    """
    agent = AutocompleteAgent(model=model)

    # Validate test case structure early to avoid cascading errors
    if not isinstance(test_case, dict) or "input" not in test_case or "ground_truth" not in test_case:
        print(f"Error in rollout {rollout_id}: invalid test_case {test_case}")
        return {
            "trajectory": None,
            "completion": None,
            "ground_truth": test_case.get("ground_truth") if isinstance(test_case, dict) else None,
            "error": "Invalid test_case: expected dict with 'input' and 'ground_truth'",
            "success": False,
        }

    try:
        # Get completion from agent
        start_time = time.time()
        completion, tool_calls, episode_info = await agent.get_completion(
            test_case["input"],
            max_turns=max_turns
        )
        latency = time.time() - start_time
        
        # Calculate reward
        reward_info = await calculate_reward(
            completion,
            test_case["ground_truth"],
            episode_info,
            tool_calls=tool_calls,
            case_metadata=test_case.get("metadata"),
            use_judge=use_judge,
            judge_model=judge_model,
        )
        
        # Build ART trajectory
        messages_and_choices = agent.get_messages_and_choices()
        conversation = agent.get_conversation()
        
        # Create metrics dictionary
        metrics = {
            "reward": reward_info["total_reward"],
            "correctness": reward_info["correctness_score"],
            "is_correct": 1.0 if reward_info["is_correct"] else 0.0,
            "num_tool_calls": episode_info["tool_calls_count"],
            "num_turns": episode_info["turns"],
            "completed": 1.0 if episode_info["completed"] else 0.0,
            "latency": latency,
            "step": step,
            "rollout_id": rollout_id,
            "used_search": reward_info.get("used_search", 0.0),
            "lookup_coverage": reward_info.get("lookup_coverage", 0.0),
        }
        
        # Create ART trajectory
        trajectory = art.Trajectory(
            messages_and_choices=messages_and_choices,
            reward=reward_info["total_reward"],
            metrics=metrics
        )

        log_record = _make_rollout_log_record(
            timestamp=datetime.utcnow().isoformat(),
            step=step,
            rollout_id=rollout_id,
            test_case=test_case,
            completion=completion,
            conversation=conversation,
            tool_calls=tool_calls,
            episode_info=episode_info,
            reward_info=reward_info,
            latency=latency,
            model=model,
            judge_model=judge_model,
        )

        return {
            "trajectory": trajectory,
            "completion": completion,
            "ground_truth": test_case["ground_truth"],
            "reward_info": reward_info,
            "episode_info": episode_info,
            "log_record": log_record,
            "success": True
        }
        
    except Exception as e:
        print(f"Error in rollout {rollout_id}: {e}")
        return {
            "trajectory": None,
            "completion": None,
            "ground_truth": test_case.get("ground_truth") if isinstance(test_case, dict) else None,
            "error": str(e),
            "success": False
        }

async def conduct_rollouts(
    model: Any,
    test_cases: List[Dict[str, str]],
    num_rollouts_per_case: int,
    step: int,
    use_judge: bool = True,
    judge_model: Optional[str] = None,
    log_path: Optional[str] = None,
) -> List[art.TrajectoryGroup]:
    """
    Conduct multiple rollouts for a set of test cases
    
    Args:
        model: ART model to use
        test_cases: List of test cases
        num_rollouts_per_case: Number of rollouts per test case
        step: Training step number
        use_judge: Whether to use LLM judge
        
    Returns:
        List of TrajectoryGroups for training
    """
    all_rollouts = []
    rollout_id = 0
    
    # Run rollouts for each test case
    for test_case in test_cases:
        case_rollouts = []
        
        # Run multiple rollouts for this test case
        tasks = []
        for _ in range(num_rollouts_per_case):
            tasks.append(
                run_single_rollout(
                    model=model,
                    test_case=test_case,
                    rollout_id=rollout_id,
                    step=step,
                    use_judge=use_judge,
                    judge_model=judge_model,
                )
            )
            rollout_id += 1
        
        # Execute rollouts concurrently
        results = await asyncio.gather(*tasks)
        
        # Collect successful trajectories
        effective_log_path = log_path or os.getenv("TRAIN_LOG_PATH")
        for result in results:
            if result["success"] and result["trajectory"]:
                case_rollouts.append(result["trajectory"])
                if effective_log_path and "log_record" in result:
                    _append_jsonl(effective_log_path, result["log_record"])
        
        if case_rollouts:
            all_rollouts.extend(case_rollouts)
    
    # Group trajectories (in this case, all trajectories are from the same model)
    if all_rollouts:
        return [art.TrajectoryGroup(all_rollouts)]
    else:
        return []

async def run_validation(
    my_model: Any,
    benchmark_model: Any,
    num_validation_cases: int = 50,
    step: int = 0,
    use_judge: bool = True,
    judge_model: Optional[str] = None,
    log_path: Optional[str] = None,
) -> List[art.Trajectory]:
    """
    Run validation against a benchmark model
    
    Args:
        my_model: Model being trained
        benchmark_model: Benchmark model (e.g., GPT-4)
        num_validation_cases: Number of validation cases
        step: Training step number
        use_judge: Whether to use LLM judge
        
    Returns:
        List of validation trajectories with win/loss rewards
    """
    # Build deterministic validation holdouts: 5 tickers, 5 metrics, 5 periods + "latest"
    try:
        all_tickers = await get_tickers_with_data()
        allowed_tickers = set(sorted(all_tickers)[:5]) if all_tickers else set()
    except Exception:
        allowed_tickers = set()

    try:
        all_metrics = await get_all_metrics()
        metric_names = sorted([m.get("metric_name") for m in all_metrics if m.get("metric_name")])
        allowed_metrics = set(metric_names[:5]) if metric_names else set()
    except Exception:
        allowed_metrics = set()

    try:
        all_periods = await get_all_periods()
        allowed_periods = set(all_periods[:5]) if all_periods else set()
    except Exception:
        allowed_periods = set()
    # Always include "latest" as allowed for validation
    allowed_periods.add("latest")

    # Log chosen holdouts (console)
    print("Validation holdouts:")
    print(f"  Tickers: {sorted(list(allowed_tickers))}")
    print(f"  Metrics: {sorted(list(allowed_metrics))}")
    print(f"  Periods: {sorted(list(allowed_periods))}")

    # Generate validation test cases using terminal distribution via curriculum_stage=3
    val_cases = await generate_cases(
        num_validation_cases,
        curriculum_stage=3,
        allowed_tickers=allowed_tickers if allowed_tickers else None,
        allowed_metrics=allowed_metrics if allowed_metrics else None,
        allowed_periods=allowed_periods if allowed_periods else None,
    )
    
    validation_trajectories = []
    
    for i, test_case in enumerate(val_cases):
        # Run both models on the same test case
        my_result = await run_single_rollout(
            model=my_model,
            test_case=test_case,
            rollout_id=i,
            step=step,
            use_judge=use_judge,
            judge_model=judge_model,
        )
        
        benchmark_result = await run_single_rollout(
            model=benchmark_model,
            test_case=test_case,
            rollout_id=i + 1000,  # Different ID space
            step=step,
            use_judge=use_judge,
            judge_model=judge_model,
        )
        
        if my_result["success"] and my_result["trajectory"]:
            # Compare rewards to determine win/loss
            my_reward = my_result["reward_info"]["total_reward"]
            benchmark_reward = benchmark_result["reward_info"]["total_reward"] if benchmark_result["success"] else 0.0
            
            # Create validation trajectory with binary reward
            val_trajectory = my_result["trajectory"]
            val_trajectory.reward = 1.0 if my_reward > benchmark_reward else 0.0
            val_trajectory.metrics["win_rate"] = val_trajectory.reward
            val_trajectory.metrics["my_reward"] = my_reward
            val_trajectory.metrics["benchmark_reward"] = benchmark_reward
            
            # Log validation record
            effective_log_path = log_path or os.getenv("VAL_LOG_PATH")
            if effective_log_path and "log_record" in my_result:
                rec = dict(my_result["log_record"])  # shallow copy
                rec["is_validation"] = True
                rec["benchmark_reward"] = benchmark_reward
                _append_jsonl(effective_log_path, rec)

            validation_trajectories.append(val_trajectory)
    
    return validation_trajectories

# ============== Batch Processing ==============

async def generate_training_trajectories(
    model: Any,
    num_cases: int = 10,
    num_rollouts_per_case: int = 5,
    step: int = 0,
    use_judge: bool = True,
    judge_model: Optional[str] = None,
) -> Tuple[List[art.TrajectoryGroup], Dict[str, float]]:
    """
    Generate training trajectories for a training step
    
    Args:
        model: ART model to train
        num_cases: Number of unique test cases
        num_rollouts_per_case: Rollouts per test case
        step: Training step number
        use_judge: Whether to use LLM judge
        
    Returns:
        Tuple of (trajectory_groups, metrics_dict)
    """
    # Determine curriculum stage from step
    # Stages: 1 for early, 2 for mid, 3 for late/default
    if step < 40:
        curriculum_stage = 1
    elif step < 80:
        curriculum_stage = 2
    else:
        curriculum_stage = 3

    # Allow override via environment variable CURRICULUM_STAGE
    import os
    env_stage = os.getenv("CURRICULUM_STAGE")
    if env_stage is not None:
        try:
            curriculum_stage = int(env_stage)
        except ValueError:
            pass

    # Generate test cases with curriculum
    test_cases = await generate_cases(num_cases, curriculum_stage=curriculum_stage)
    
    # Conduct rollouts
    trajectory_groups = await conduct_rollouts(
        model=model,
        test_cases=test_cases,
        num_rollouts_per_case=num_rollouts_per_case,
        step=step,
        use_judge=use_judge,
        judge_model=judge_model,
    )
    
    # Calculate metrics
    total_trajectories = sum(len(tg.trajectories) for tg in trajectory_groups)
    avg_reward = 0.0
    avg_correctness = 0.0
    avg_tool_calls = 0.0
    
    if total_trajectories > 0:
        for tg in trajectory_groups:
            for traj in tg.trajectories:
                avg_reward += traj.metrics.get("reward", 0.0)
                avg_correctness += traj.metrics.get("correctness", 0.0)
                avg_tool_calls += traj.metrics.get("num_tool_calls", 0.0)
        
        avg_reward /= total_trajectories
        avg_correctness /= total_trajectories
        avg_tool_calls /= total_trajectories
    
    metrics = {
        "total_trajectories": total_trajectories,
        "avg_reward": avg_reward,
        "avg_correctness": avg_correctness,
        "avg_tool_calls": avg_tool_calls,
        "num_groups": len(trajectory_groups)
    }
    
    return trajectory_groups, metrics

# ============== Testing ==============

if __name__ == "__main__":
    async def test_rollout():
        from database import setup_database
        
        # Setup database
        await setup_database() # Requires TIINGO_API_KEY
        
        # Create a mock model that just returns OpenAI responses
        class MockModel:
            name = "test-model"
            
            async def __call__(self, messages):
                # This would normally call the model
                return "return_answer(answer='$383.3 billion')"
        
        model = MockModel()
        
        # Test single rollout
        test_case = {
            "input": "Apple's revenue in 2023 was ",
            "ground_truth": "$383.3 billion"
        }
        
        print("Testing single rollout...")
        result = await run_single_rollout(
            model=model,
            test_case=test_case,
            rollout_id=0,
            step=0,
            use_judge=False  # Don't use judge for testing
        )
        
        if result["success"]:
            print(f"Completion: {result['completion']}")
            print(f"Reward: {result['reward_info']['total_reward']:.3f}")
            print(f"Correct: {result['reward_info']['is_correct']}")
        else:
            print(f"Error: {result.get('error')}")
        
        # Test batch rollouts
        print("\nTesting batch rollouts...")
        test_cases = [
            {"input": "Microsoft's revenue in 2023 was ", "ground_truth": "$211.9 billion"},
            {"input": "The CFO said that ", "ground_truth": "NO_COMPLETION_NEEDED"}
        ]
        
        trajectory_groups = await conduct_rollouts(
            model=model,
            test_cases=test_cases,
            num_rollouts_per_case=2,
            step=0,
            use_judge=False
        )
        
        print(f"Generated {len(trajectory_groups)} trajectory groups")
        if trajectory_groups:
            print(f"First group has {len(trajectory_groups[0].trajectories)} trajectories")
    
    asyncio.run(test_rollout())