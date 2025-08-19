"""
Minimal FastAPI server for testing the autocomplete training modules
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio
import os
import time

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import our modules
from database import setup_database, get_db
from synthetic import generate_cases
from agent import AutocompleteAgent
from rewards import calculate_reward

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to replace deprecated startup/shutdown events."""
    import sys
    print("Initializing database...")
    try:
        force_reload = getattr(app.state, 'force_reload', False)
        await setup_database(force_reload=force_reload)  # Requires TIINGO_API_KEY
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("\nPlease set TIINGO_API_KEY in your environment or .env file", file=sys.stderr)
        sys.exit(1)
    print("Database ready")
    # Startup complete
    yield
    # Shutdown logic could go here if needed

app = FastAPI(title="Finance Autocomplete Test Server", lifespan=lifespan)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class EvaluationRequest(BaseModel):
    models: List[str] = ["gpt-4.1-mini"]
    num_cases: int = 10
    use_judge: bool = True
    # Optional: control prompt phrasing used by synthetic case generator
    # Values: None/"default" for training-style prompts, "validation" for alt phrasing
    templates_mode: Optional[str] = None

class EvaluationResponse(BaseModel):
    results: List[Dict[str, Any]]
    accuracy_scores: Dict[str, float]
    # Allow metadata objects in test cases
    test_cases: List[Dict[str, Any]]

# Live autocomplete request/response models
class AutocompleteRequest(BaseModel):
    text: str
    model: Optional[str] = None
    max_turns: int = 7

class AutocompleteResponse(BaseModel):
    completion: Optional[str]
    used_model: str
    latency_sec: float
    tool_calls: int

# Startup handled via lifespan

# Endpoints
@app.get("/")
async def serve_frontend():
    """Serve the HTML frontend"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {"message": "Frontend not found. Please create index.html"}

@app.get("/api/financial-data")
async def get_financial_data():
    """Get all financial data from the database"""
    async with get_db() as db:
        # Get all data with ticker and metric info
        query = """
        SELECT 
            fd.ticker,
            t.company_name,
            fd.metric_name,
            m.description as metric_description,
            fd.period,
            fd.value,
            fd.unit
        FROM financial_data fd
        JOIN tickers t ON fd.ticker = t.ticker
        JOIN metrics m ON fd.metric_name = m.metric_name
        ORDER BY fd.ticker, fd.period DESC, fd.metric_name
        """
        
        async with db.execute(query) as cursor:
            rows = await cursor.fetchall()
            data = []
            for row in rows:
                data.append({
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "metric": row["metric_name"],
                    "metric_description": row["metric_description"],
                    "period": row["period"],
                    "value": row["value"],
                    "unit": row["unit"]
                })
            
            return {"data": data, "total": len(data)}

@app.post("/api/batch-evaluation", response_model=EvaluationResponse)
async def batch_evaluation(request: EvaluationRequest):
    """
    Run batch evaluation on specified models
    """
    # Generate test cases
    test_cases = await generate_cases(
        request.num_cases,
        templates_mode=(request.templates_mode or None)
    )
    # Defensive filter: ensure valid structure
    test_cases = [tc for tc in test_cases if isinstance(tc, dict) and "input" in tc and "ground_truth" in tc]
    
    if not test_cases:
        raise HTTPException(status_code=500, detail="Failed to generate test cases")
    
    results = []
    model_correct_counts = {model: 0 for model in request.models}
    
    # Test each case with each model
    for i, test_case in enumerate(test_cases):
        case_result = {
            "case_id": i,
            "input": test_case.get("input"),
            "ground_truth": test_case.get("ground_truth"),
            "model_results": {}
        }
        
        for model_name in request.models:
            try:
                # Create agent with specified model
                if model_name.startswith("gpt"):
                    # Use OpenAI models
                    from openai import AsyncOpenAI
                    
                    # Create a simple model wrapper
                    class OpenAIModel:
                        def __init__(self, model_name):
                            self.name = model_name
                            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        
                        async def __call__(self, messages):
                            response = await self.client.chat.completions.create(
                                model=self.name,
                                messages=messages,
                                temperature=0.7,
                                top_p=0.9,
                                max_tokens=128
                            )
                            return response.choices[0].message.content.strip()
                    
                    model = OpenAIModel(model_name)
                elif model_name.startswith("ollama:"):
                    # Use local Ollama models
                    # Model name format: "ollama:<model_id>", e.g., "ollama:llama3"
                    class OllamaModel:
                        def __init__(self, model_name):
                            # Strip prefix
                            self.name = model_name.split(":", 1)[1]
                        async def __call__(self, messages):
                            try:
                                import asyncio
                                import ollama
                                loop = asyncio.get_running_loop()
                                def _chat():
                                    return ollama.chat(
                                        model=self.name,
                                        messages=messages,
                                        options={"temperature": 0.7, "top_p": 0.9, "num_predict": 128}
                                    )
                                response = await loop.run_in_executor(None, _chat)
                                return response["message"]["content"].strip()
                            except Exception as e:
                                return f"return_answer(answer='[ollama error: {str(e)}]')"
                    model = OllamaModel(model_name)
                else:
                    # Default to a mock model for testing
                    class MockModel:
                        name = model_name
                        async def __call__(self, messages):
                            return "return_answer(answer='test response')"
                    model = MockModel()
                
                # Create agent and get completion
                agent = AutocompleteAgent(model=model)
                inference_start = time.perf_counter()
                completion, tool_calls, episode_info = await agent.get_completion(
                    test_case.get("input", ""),
                    max_turns=7
                )
                inference_end = time.perf_counter()
                
                # Calculate reward/correctness
                reward_info = await calculate_reward(
                    completion,
                    test_case.get("ground_truth", ""),
                    episode_info,
                    tool_calls=tool_calls,
                    case_metadata=test_case.get("metadata"),
                    use_judge=request.use_judge
                )
                
                # Store result
                case_result["model_results"][model_name] = {
                    "prediction": completion,
                    "is_correct": reward_info["is_correct"],
                    "reward": reward_info["total_reward"],
                    "judge_score": reward_info.get("correctness_score", None),
                    "tool_calls": episode_info["tool_calls_count"],
                    "tool_calls_log": tool_calls,  # Add full tool call log
                    "turns": episode_info.get("turns", None),
                    "conversation": agent.get_conversation(),
                    "reasoning": reward_info.get("reasoning", ""),
                    "used_search": reward_info.get("used_search", 0.0),
                    "lookup_coverage": reward_info.get("lookup_coverage", 0.0),
                    "coverage_bonus": reward_info.get("coverage_bonus", 0.0),
                    "ticker_correct": reward_info.get("ticker_correct", 0.0),
                    "metric_correct": reward_info.get("metric_correct", 0.0),
                    "period_correct": reward_info.get("period_correct", 0.0),
                    "exact_tuple_match": reward_info.get("exact_tuple_match", 0.0),
                    # Negative reward components
                    "total_penalty": reward_info.get("total_penalty", 0.0),
                    "char_penalty": reward_info.get("char_penalty", 0.0),
                    "toolcalls_per_turn_penalty": reward_info.get("toolcalls_per_turn_penalty", 0.0),
                    "turns_penalty": reward_info.get("turns_penalty", 0.0),
                    "format_penalty_applied": reward_info.get("format_penalty_applied", 0.0),
                    "invalid_turns_count": reward_info.get("invalid_turns_count", 0),
                    "searched_when_no_completion": reward_info.get("searched_when_no_completion", 0.0),
                    "incorrect_abstain": reward_info.get("incorrect_abstain", 0.0),
                    # Telemetry (optional for UI)
                    "telemetry": reward_info.get("telemetry", {}),
                    # Latency metrics (seconds)
                    "latency_sec": (inference_end - inference_start),
                }
                
                if reward_info["is_correct"]:
                    model_correct_counts[model_name] += 1
                    
            except Exception as e:
                print(f"Error evaluating {model_name} on case {i}: {e}")
                case_result["model_results"][model_name] = {
                    "prediction": None,
                    "is_correct": False,
                    "error": str(e)
                }
        
        results.append(case_result)
    
    # Calculate accuracy scores
    accuracy_scores = {
        model: (count / len(test_cases)) if test_cases else 0.0
        for model, count in model_correct_counts.items()
    }
    
    return EvaluationResponse(
        results=results,
        accuracy_scores=accuracy_scores,
        test_cases=test_cases
    )

@app.post("/api/autocomplete", response_model=AutocompleteResponse)
async def live_autocomplete(request: AutocompleteRequest):
    """Single-turn live autocomplete for a given input prefix.
    Returns a suggested completion if the model deems one is needed.
    """
    model_name = request.model or os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")

    # Build model (reuse simple wrappers from batch evaluator)
    if model_name.startswith("gpt"):
        from openai import AsyncOpenAI

        class OpenAIModel:
            def __init__(self, name: str):
                self.name = name
                self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            async def __call__(self, messages):
                response = await self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=64,
                )
                return response.choices[0].message.content.strip()

        model = OpenAIModel(model_name)
    elif model_name.startswith("ollama:"):
        class OllamaModel:
            def __init__(self, name: str):
                self.name = name.split(":", 1)[1]

            async def __call__(self, messages):
                try:
                    import asyncio
                    import ollama
                    loop = asyncio.get_running_loop()

                    def _chat():
                        return ollama.chat(
                            model=self.name,
                            messages=messages,
                            options={"temperature": 0.7, "top_p": 0.9, "num_predict": 64},
                        )

                    response = await loop.run_in_executor(None, _chat)
                    return response["message"]["content"].strip()
                except Exception as e:
                    return f"return_answer(answer='[ollama error: {str(e)}]')"

        model = OllamaModel(model_name)
    else:
        class MockModel:
            name = model_name

            async def __call__(self, messages):
                return "return_answer(answer='')"

        model = MockModel()

    # Run the agent once for this input
    agent = AutocompleteAgent(model=model)
    start = time.perf_counter()
    completion, _tool_calls, episode_info = await agent.get_completion(
        request.text, max_turns=max(1, int(request.max_turns))
    )
    end = time.perf_counter()

    # Normalize completion: treat empty/placeholder as no suggestion
    normalized = (completion or "").strip() if isinstance(completion, str) else None
    if not normalized or normalized == "NO_COMPLETION_NEEDED":
        normalized = None

    return AutocompleteResponse(
        completion=normalized,
        used_model=model_name,
        latency_sec=(end - start),
        tool_calls=int(episode_info.get("tool_calls_count", 0)),
    )

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description='Finance Autocomplete Test Server')
    parser.add_argument('--reload-data', action='store_true', 
                        help='Force reload data from Tiingo API even if database exists')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Port to run server on (default: 8000)')
    args = parser.parse_args()
    
    # Store reload flag in app state for startup event
    app.state.force_reload = args.reload_data
    
    print(f"Starting server at http://localhost:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser to use the frontend")
    if args.reload_data:
        print("Force reloading data from Tiingo API...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)