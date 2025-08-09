"""
Financial tool execution environment for RL training
Handles tool calls and maintains state for multi-turn interactions
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
from database import (
    get_all_tickers, get_tickers_with_names, get_all_metrics,
    get_financial_value, get_available_periods, get_latest_period
)

class ToolName(Enum):
    """Available tools in the financial environment"""
    GET_METRICS = "get_metrics"
    GET_TICKERS = "get_tickers"
    GET_VALUE = "get_value"
    CALCULATE = "calculate"
    RETURN_ANSWER = "return_answer"

class ToolCall:
    """Represents a single tool call with arguments and result"""
    def __init__(self, tool_name: str, arguments: Dict[str, Any], result: Any = None):
        self.tool_name = tool_name
        self.arguments = arguments
        self.result = result
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "timestamp": self.timestamp
        }

class FinancialEnvironment:
    """
    Environment that executes financial tool calls
    Maintains state across multi-turn interactions
    """
    
    def __init__(self):
        self.tool_calls: List[ToolCall] = []
        self.episode_complete = False
        self.final_answer = None
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool call and return the result
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Result of the tool execution
        """
        result = None
        
        if tool_name == ToolName.GET_METRICS.value:
            result = await self._get_metrics()
        elif tool_name == ToolName.GET_TICKERS.value:
            result = await self._get_tickers()
        elif tool_name == ToolName.GET_VALUE.value:
            result = await self._get_value(**arguments)
        elif tool_name == ToolName.CALCULATE.value:
            result = self._calculate(**arguments)
        elif tool_name == ToolName.RETURN_ANSWER.value:
            result = self._return_answer(**arguments)
            self.episode_complete = True
            self.final_answer = result
        
        # Record the tool call
        tool_call = ToolCall(tool_name, arguments, result)
        self.tool_calls.append(tool_call)
        
        return result
    
    async def _get_metrics(self) -> List[Dict[str, str]]:
        """Get all available financial metrics"""
        metrics = await get_all_metrics()
        return [
            {
                "name": m["metric_name"],
                "description": m["description"],
                "unit": m["unit"]
            }
            for m in metrics
        ]
    
    async def _get_tickers(self) -> List[Dict[str, str]]:
        """Get all available stock tickers with company names"""
        return await get_tickers_with_names()
    
    async def _get_value(self, metric: str, ticker: str, period: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific financial value
        
        Args:
            metric: Metric name (e.g., "revenue")
            ticker: Stock ticker (e.g., "AAPL")
            period: Time period (e.g., "2023Q4" or "latest")
        """
        # Handle "latest" period request
        if period.lower() == "latest":
            actual_period = await get_latest_period(ticker, metric)
            if actual_period:
                period = actual_period
            else:
                return None
        
        value_data = await get_financial_value(ticker, metric, period)
        
        if value_data:
            # Add the period to the result for clarity
            value_data["period"] = period
            value_data["ticker"] = ticker
            value_data["metric"] = metric
        
        return value_data
    
    def _calculate(
        self,
        num1: float,
        num2: float,
        operation: str,
        duration: Optional[int] = None
    ) -> Optional[float]:
        """
        Perform a calculation
        
        Args:
            num1: First number
            num2: Second number
            operation: Operation to perform (add, subtract, multiply, divide, CAGR)
            duration: Duration for CAGR calculation (number of periods)
        """
        result = None
        
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            result = num1 / num2 if num2 != 0 else None
        elif operation == "CAGR" and duration:
            # CAGR calculation: ((final/initial)^(1/duration) - 1) * 100
            # Allow CAGR when ratio is positive (handles negative values)
            if num2 != 0 and duration > 0 and (num1 / num2) > 0:
                result = ((num1 / num2) ** (1 / duration) - 1) * 100
        
        return result
    
    def _return_answer(self, answer: str) -> str:
        """
        Return the final answer/completion
        
        Args:
            answer: The completion text to return
        """
        return answer
    
    def reset(self):
        """Reset the environment for a new episode"""
        self.tool_calls = []
        self.episode_complete = False
        self.final_answer = None
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment
        
        Returns:
            Dictionary containing tool calls, completion status, and final answer
        """
        return {
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "episode_complete": self.episode_complete,
            "final_answer": self.final_answer,
            "num_tool_calls": len(self.tool_calls)
        }
    
    def get_tool_calls_log(self) -> List[Dict[str, Any]]:
        """Get a log of all tool calls made"""
        return [tc.to_dict() for tc in self.tool_calls]
    
    def is_complete(self) -> bool:
        """Check if the episode is complete"""
        return self.episode_complete
    
    def get_final_answer(self) -> Optional[str]:
        """Get the final answer if episode is complete"""
        return self.final_answer

class EnvironmentError(Exception):
    """Custom exception for environment errors"""
    pass

# ============== Tool Call Parser ==============

import re

def parse_tool_calls_from_response(response: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from LLM response text
    
    Args:
        response: Raw LLM response containing tool calls
        
    Returns:
        List of parsed tool calls with tool name and arguments
    """
    tool_calls = []
    
    # Define function signatures and their expected parameters
    tool_signatures = {
        "get_metrics": [],
        "get_tickers": [],
        "get_value": ["metric", "ticker", "period"],
        "calculate": ["num1", "num2", "operation", "duration"],
        "return_answer": ["answer"]
    }
    
    # Find the first tool call in the response
    earliest_match = None
    earliest_pos = len(response)
    matched_tool = None
    
    # Find which tool appears first in the response
    for tool_name, params in tool_signatures.items():
        pattern = rf"{tool_name}\s*\("
        match = re.search(pattern, response)
        if match and match.start() < earliest_pos:
            earliest_match = match
            earliest_pos = match.start()
            matched_tool = (tool_name, params)
    
    # If we found a tool call, parse only that one
    if earliest_match and matched_tool:
        tool_name, params = matched_tool
        match = earliest_match
        start = match.end()
        
        # Find matching closing parenthesis
        paren_count = 1
        end = start
        while end < len(response) and paren_count > 0:
            if response[end] == '(':
                paren_count += 1
            elif response[end] == ')':
                paren_count -= 1
            end += 1
        
        if paren_count == 0:
            args_str = response[start:end-1].strip()
            
            # Parse arguments
            parsed_args = {}
            
            if not args_str:  # No arguments
                if tool_name in ["get_metrics", "get_tickers"]:
                    tool_calls.append({"tool": tool_name, "args": {}})
            else:
                # Special handling for return_answer
                if tool_name == "return_answer":
                    answer = args_str
                    if answer.startswith('answer='):
                        answer = answer[7:].strip()
                    # Remove surrounding quotes
                    for quote in ['"', "'", '"""', "'''"]:
                        if answer.startswith(quote) and answer.endswith(quote):
                            answer = answer[len(quote):-len(quote)]
                            break
                    parsed_args = {"answer": answer}
                else:
                    # Parse other functions
                    has_keywords = '=' in args_str
                    
                    if has_keywords:
                        # Parse keyword arguments
                        parts = re.split(r',(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', args_str)
                        for part in parts:
                            if '=' in part:
                                key, val = part.split('=', 1)
                                key = key.strip()
                                val = val.strip()
                                # Remove quotes
                                for quote in ['"', "'", '"""', "'''"]:
                                    if val.startswith(quote) and val.endswith(quote):
                                        val = val[len(quote):-len(quote)]
                                        break
                                if key in params:
                                    parsed_args[key] = val
                    else:
                        # Parse positional arguments
                        parts = re.split(r',(?=(?:[^"\']*["\'][^"\']*["\'])*[^"\']*$)', args_str)
                        for i, part in enumerate(parts):
                            if i < len(params):
                                val = part.strip()
                                # Remove quotes
                                for quote in ['"', "'", '"""', "'''"]:
                                    if val.startswith(quote) and val.endswith(quote):
                                        val = val[len(quote):-len(quote)]
                                        break
                                parsed_args[params[i]] = val
                
                # Convert types for calculate function
                if tool_name == "calculate" and parsed_args:
                    if "num1" in parsed_args:
                        parsed_args["num1"] = float(parsed_args["num1"])
                    if "num2" in parsed_args:
                        parsed_args["num2"] = float(parsed_args["num2"])
                    if "duration" in parsed_args:
                        duration_val = parsed_args["duration"]
                        if duration_val and duration_val.lower() != 'none':
                            try:
                                parsed_args["duration"] = int(duration_val)
                            except ValueError:
                                parsed_args["duration"] = None
                        else:
                            parsed_args["duration"] = None
                
                if parsed_args:
                    tool_calls.append({"tool": tool_name, "args": parsed_args})
    
    return tool_calls

# ============== Testing ==============

if __name__ == "__main__":
    import asyncio
    
    async def test_environment():
        from database import setup_database
        
        # Setup database
        await setup_database()  # Requires TIINGO_API_KEY
        
        # Create environment
        env = FinancialEnvironment()
        
        # Test tool executions
        print("Testing Financial Environment...")
        
        # Get metrics
        metrics = await env.execute_tool("get_metrics", {})
        print(f"Found {len(metrics)} metrics")
        
        # Get tickers
        tickers = await env.execute_tool("get_tickers", {})
        print(f"Found {len(tickers)} tickers")
        
        # Get a value
        if tickers and metrics:
            value = await env.execute_tool(
                "get_value",
                {"metric": "revenue", "ticker": "AAPL", "period": "2023FY"}
            )
            print(f"AAPL 2023FY revenue: {value}")
        
        # Calculate
        result = await env.execute_tool(
            "calculate",
            {"num1": 100, "num2": 80, "operation": "subtract", "duration": None}
        )
        print(f"100 - 80 = {result}")
        
        # Return answer
        answer = await env.execute_tool(
            "return_answer",
            {"answer": "$383.3 billion"}
        )
        print(f"Final answer: {answer}")
        
        # Check state
        state = env.get_state()
        print(f"\nEnvironment state:")
        print(f"- Complete: {state['episode_complete']}")
        print(f"- Tool calls: {state['num_tool_calls']}")
        print(f"- Final answer: {state['final_answer']}")
        
        # Test parser
        print("\nTesting tool call parser...")
        test_response = 'get_value(metric="revenue", ticker="AAPL", period="2023FY")'
        parsed = parse_tool_calls_from_response(test_response)
        print(f"Parsed: {parsed}")
    
    asyncio.run(test_environment())