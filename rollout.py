"""
Rollout module for generating autocomplete trajectories
Handles episode generation and trajectory building for ART training
"""

import asyncio
import time
import uuid
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

async def run_single_rollout(
    model: Any,
    test_case: Dict[str, str],
    rollout_id: int,
    step: int,
    use_judge: bool = True,
    max_turns: int = 10
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
            use_judge=use_judge
        )
        
        # Build ART trajectory
        messages_and_choices = agent.get_messages_and_choices()
        
        # Create metrics dictionary
        metrics = {
            "reward": reward_info["total_reward"],
            "correctness": reward_info["correctness_score"],
            "efficiency_bonus": reward_info["efficiency_bonus"],
            "completion_penalty": reward_info["completion_penalty"],
            "is_correct": 1.0 if reward_info["is_correct"] else 0.0,
            "num_tool_calls": episode_info["tool_calls_count"],
            "num_turns": episode_info["turns"],
            "completed": 1.0 if episode_info["completed"] else 0.0,
            "latency": latency,
            "step": step,
            "rollout_id": rollout_id
        }
        
        # Create ART trajectory
        trajectory = art.Trajectory(
            messages_and_choices=messages_and_choices,
            reward=reward_info["total_reward"],
            metrics=metrics
        )

        
        return {
            "trajectory": trajectory,
            "completion": completion,
            "ground_truth": test_case["ground_truth"],
            "reward_info": reward_info,
            "episode_info": episode_info,
            "success": True
        }
        
    except Exception as e:
        print(f"Error in rollout {rollout_id}: {e}")
        return {
            "trajectory": None,
            "completion": None,
            "ground_truth": test_case["ground_truth"],
            "error": str(e),
            "success": False
        }

async def conduct_rollouts(
    model: Any,
    test_cases: List[Dict[str, str]],
    num_rollouts_per_case: int,
    step: int,
    use_judge: bool = True
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
                    use_judge=use_judge
                )
            )
            rollout_id += 1
        
        # Execute rollouts concurrently
        results = await asyncio.gather(*tasks)
        
        # Collect successful trajectories
        for result in results:
            if result["success"] and result["trajectory"]:
                case_rollouts.append(result["trajectory"])
        
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
    use_judge: bool = True
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
    # Generate validation test cases
    val_cases = await generate_cases(num_validation_cases)
    
    validation_trajectories = []
    
    for i, test_case in enumerate(val_cases):
        # Run both models on the same test case
        my_result = await run_single_rollout(
            model=my_model,
            test_case=test_case,
            rollout_id=i,
            step=step,
            use_judge=use_judge
        )
        
        benchmark_result = await run_single_rollout(
            model=benchmark_model,
            test_case=test_case,
            rollout_id=i + 1000,  # Different ID space
            step=step,
            use_judge=use_judge
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
            
            validation_trajectories.append(val_trajectory)
    
    return validation_trajectories

# ============== Batch Processing ==============

async def generate_training_trajectories(
    model: Any,
    num_cases: int = 10,
    num_rollouts_per_case: int = 5,
    step: int = 0,
    use_judge: bool = True
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
    # Generate test cases
    test_cases = await generate_cases(num_cases)
    
    # Conduct rollouts
    trajectory_groups = await conduct_rollouts(
        model=model,
        test_cases=test_cases,
        num_rollouts_per_case=num_rollouts_per_case,
        step=step,
        use_judge=use_judge
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