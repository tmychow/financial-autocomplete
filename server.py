"""
Minimal FastAPI server for testing the autocomplete training modules
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import our modules
from database import setup_database, get_db
from synthetic import generate_cases
from agent import AutocompleteAgent
from rewards import calculate_reward

app = FastAPI(title="Finance Autocomplete Test Server")

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

class EvaluationResponse(BaseModel):
    results: List[Dict[str, Any]]
    accuracy_scores: Dict[str, float]
    test_cases: List[Dict[str, str]]

# Startup event to initialize database
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    import sys
    print("Initializing database...")
    try:
        # Check if force_reload flag was set via command line
        force_reload = getattr(app.state, 'force_reload', False)
        await setup_database(force_reload=force_reload)  # Requires TIINGO_API_KEY
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("\nPlease set TIINGO_API_KEY in your environment or .env file", file=sys.stderr)
        sys.exit(1)
    print("Database ready")

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
    test_cases = await generate_cases(request.num_cases)
    
    if not test_cases:
        raise HTTPException(status_code=500, detail="Failed to generate test cases")
    
    results = []
    model_correct_counts = {model: 0 for model in request.models}
    
    # Test each case with each model
    for i, test_case in enumerate(test_cases):
        case_result = {
            "case_id": i,
            "input": test_case["input"],
            "ground_truth": test_case["ground_truth"],
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
                                temperature=0.1,
                                max_tokens=500
                            )
                            return response.choices[0].message.content.strip()
                    
                    model = OpenAIModel(model_name)
                else:
                    # Default to a mock model for testing
                    class MockModel:
                        name = model_name
                        async def __call__(self, messages):
                            return "return_answer(answer='test response')"
                    model = MockModel()
                
                # Create agent and get completion
                agent = AutocompleteAgent(model=model)
                completion, tool_calls, episode_info = await agent.get_completion(
                    test_case["input"],
                    max_turns=10
                )
                
                # Calculate reward/correctness
                reward_info = await calculate_reward(
                    completion,
                    test_case["ground_truth"],
                    episode_info,
                    use_judge=request.use_judge
                )
                
                # Store result
                case_result["model_results"][model_name] = {
                    "prediction": completion,
                    "is_correct": reward_info["is_correct"],
                    "reward": reward_info["total_reward"],
                    "tool_calls": episode_info["tool_calls_count"],
                    "tool_calls_log": tool_calls,  # Add full tool call log
                    "reasoning": reward_info.get("reasoning", "")
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