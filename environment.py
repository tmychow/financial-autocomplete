"""
Financial tool execution environment for RL training
Handles tool calls and maintains state for multi-turn interactions
"""

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
from database import (
    get_tickers_with_names, get_all_metrics,
    get_financial_value, get_available_periods, get_latest_period
)
from textmatch import (
    build_ticker_alias_map, build_metric_alias_map,
    match_alias, clean_company_string
)

class ToolName(Enum):
    """Available tools in the financial environment"""
    SEARCH = "search"
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
        # Lazy-loaded alias maps
        self._ticker_alias_map: Optional[Dict[str, str]] = None
        self._metric_alias_map: Optional[Dict[str, str]] = None
        
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
        
        if tool_name == ToolName.SEARCH.value:
            # Require all args before calling
            required = ["metric", "ticker", "period"]
            if not all(k in arguments and isinstance(arguments[k], str) and arguments[k].strip() for k in required):
                result = "Invalid tool usage: search expects metric, ticker, and period."
            else:
                result = await self._search(**arguments)
        elif tool_name == ToolName.CALCULATE.value:
            # Require num1, num2, operation before calling
            required = ["num1", "num2", "operation"]
            if not all(k in arguments for k in required):
                result = "Invalid tool usage: calculate expects num1, num2, and operation."
            else:
                result = self._calculate(**arguments)
        elif tool_name == ToolName.RETURN_ANSWER.value:
            if "answer" not in arguments:
                result = "Invalid tool usage: return_answer expects answer."
            else:
                result = self._return_answer(**arguments)
                self.episode_complete = True
                self.final_answer = result
        
        # Record the tool call
        tool_call = ToolCall(tool_name, arguments, result)
        self.tool_calls.append(tool_call)
        
        return result
    
    async def _search(self, metric: str, ticker: str, period: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific financial value
        
        Args:
            metric: Metric name (e.g., "revenue")
            ticker: Stock ticker (e.g., "AAPL")
            period: Time period (e.g., "2023Q4" or "latest")
        """
        # Normalize inputs
        norm_ticker = await self._normalize_ticker(ticker)
        if not norm_ticker:
            return f"Invalid ticker: {ticker}"

        norm_metric = await self._normalize_metric(metric)
        if not norm_metric:
            return f"Invalid metric: {metric}"

        norm_period = await self._normalize_period(period, norm_ticker, norm_metric)
        if not norm_period:
            return f"Invalid period: {period}"

        # Resolve 'latest' to concrete period if needed (normalize_period may already do this)
        if norm_period.lower() == "latest":
            actual_period = await get_latest_period(norm_ticker, norm_metric)
            if actual_period:
                norm_period = actual_period
            else:
                return None

        value_data = await get_financial_value(norm_ticker, norm_metric, norm_period)
        
        if value_data:
            # Add the period to the result for clarity
            value_data["period"] = norm_period
            value_data["ticker"] = norm_ticker
            value_data["metric"] = norm_metric
        
        return value_data

    async def _ensure_alias_maps(self):
        """Load alias maps for tickers and metrics on first use."""
        if self._ticker_alias_map is None:
            # Build ticker alias map from DB rows (algorithmic cleaning only)
            try:
                rows = await get_tickers_with_names()
            except Exception:
                rows = []
            self._ticker_alias_map = build_ticker_alias_map(rows)

        if self._metric_alias_map is None:
            # Build metric alias map directly from DB codes/descriptions (no hand synonyms)
            try:
                metrics = await get_all_metrics()
            except Exception:
                metrics = []
            self._metric_alias_map = build_metric_alias_map(metrics)

    async def _normalize_ticker(self, raw: str) -> Optional[str]:
        await self._ensure_alias_maps()
        if not raw:
            return None
        s = raw.strip().lower()
        # First, try exact/normalized/fuzzy via textmatch
        match = match_alias(s, self._ticker_alias_map, cutoff=0.92)
        if match:
            return match
        # Conservative substring fallback on cleaned names
        s_clean = clean_company_string(s)
        best = None
        for alias, tkr in self._ticker_alias_map.items():
            if alias != tkr.lower() and s_clean and s_clean in alias:
                best = tkr
                break
        return best

    async def _normalize_metric(self, raw: str) -> Optional[str]:
        await self._ensure_alias_maps()
        if not raw:
            return None
        s = raw.strip().lower()
        # exact/normalized/fuzzy via textmatch against DB-driven aliases
        return match_alias(s, self._metric_alias_map, cutoff=0.92)

    async def _normalize_period(self, raw: str, ticker: str, metric: str) -> Optional[str]:
        if not raw:
            return None
        s = raw.strip().lower()
        if s in {"latest", "most recent"}:
            return "latest"
        # FY patterns
        import re as _re
        # Allow separators like space/underscore/hyphen by compacting them away for matching
        s_compact = _re.sub(r"[\s_\-]+", "", s)
        # FY: accept FY2023, 2023FY, FY_2023, 2023_FY, etc.
        fy_match = _re.fullmatch(r"(fy)?(\d{4})(fy)?", s_compact)
        if fy_match:
            year = fy_match.group(2)
            return f"{year}FY"
        # Quarter patterns like Q4 2023, 2023 Q4, 2023Q4
        # Accept Q3 2023, Q3_2023, Q3-2023, 2023 Q3, 2023_Q3, 2023-Q3, and compact forms
        q_match = _re.fullmatch(r"q([1-4])(\d{4})", s_compact)
        if q_match:
            q = q_match.group(1)
            year = q_match.group(2)
            return f"{year}Q{q}"
        q_match2 = _re.fullmatch(r"(\d{4})q([1-4])", s_compact)
        if q_match2:
            year = q_match2.group(1)
            q = q_match2.group(2)
            return f"{year}Q{q}"
        # As a convenience, also handle the original spaced form (e.g., "q3 2023")
        q_match3 = _re.fullmatch(r"q([1-4])\s*(\d{4})", s)
        if q_match3:
            q = q_match3.group(1)
            year = q_match3.group(2)
            return f"{year}Q{q}"
        q_match4 = _re.fullmatch(r"(\d{4})\s*q([1-4])", s)
        if q_match4:
            year = q_match4.group(1)
            q = q_match4.group(2)
            return f"{year}Q{q}"
        # As a fallback, try to match by year and optional quarter tokens within available periods
        try:
            periods = await get_available_periods(ticker, metric)
        except Exception:
            periods = []
        # Tokenize more flexibly: treat common separators as spaces
        tokens = _re.sub(r"[,_/\-]+", " ", s).split()
        year_tokens = [t for t in tokens if t.isdigit() and len(t) == 4]
        quarter_tokens = [t for t in tokens if t.lower() in {"q1", "q2", "q3", "q4", "1", "2", "3", "4"}]

        # If the user clearly tried to specify a quarter/FY but it's malformed, treat as invalid
        has_q_like = any(t.lower().startswith('q') for t in tokens)
        has_fy_like = any(t.lower().startswith('fy') for t in tokens)
        if has_q_like and not quarter_tokens:
            return None
        # If FY-like token exists but earlier FY regex didn't match, treat as invalid rather than falling back
        if has_fy_like and not _re.fullmatch(r"(fy)?(\d{4})(fy)?", s_compact):
            return None

        # Require at least a 4-digit year to use fallback; otherwise, treat as invalid
        if not year_tokens:
            return None

        year = year_tokens[0]

        # Helper to filter periods by year
        periods_for_year = [p for p in periods if year in p]
        if not periods_for_year:
            return None

        # If quarter is specified, require an exact year+quarter match
        if quarter_tokens:
            # Normalize quarter token to Qn
            qt = None
            for qt_raw in quarter_tokens:
                if qt_raw.lower().startswith('q') and len(qt_raw) == 2 and qt_raw[1] in '1234':
                    qt = qt_raw[1]
                    break
                if qt_raw in {"1", "2", "3", "4"}:
                    qt = qt_raw
                    break
            if qt is None:
                return None
            target = f"{year}Q{qt}"
            for p in periods_for_year:
                if p == target:
                    return p
            # If exact not found, invalid
            return None

        # No quarter specified: prefer FY for that year if present, else latest quarter within that year
        fy = f"{year}FY"
        if fy in periods_for_year:
            return fy
        # Choose the highest quarter number available for that year
        best_q = 0
        best_period = None
        for p in periods_for_year:
            m = _re.fullmatch(rf"{year}Q([1-4])", p)
            if m:
                qn = int(m.group(1))
                if qn > best_q:
                    best_q = qn
                    best_period = p
        return best_period
    
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
            operation: Operation to perform (add, subtract, multiply, divide)
        """
        result = None

        # Validate numeric inputs; if invalid, return a simple error string
        if not isinstance(num1, (int, float)) or not isinstance(num2, (int, float)):
            return "Invalid tool usage: calculate expects numeric num1 and num2."
        
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            result = num1 / num2 if num2 != 0 else None
        
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
    
    def _is_strict_number(value: Any) -> bool:
        """Return True only for plain numeric literals (ints/floats as types, or numeric-looking strings)."""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            return re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", value) is not None
        return False
    
    # Define function signatures and their expected parameters
    tool_signatures = {
        "search": ["metric", "ticker", "period"],
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
            
            if args_str or tool_name == "return_answer":
                # Special handling for return_answer
                if tool_name == "return_answer":
                    answer = args_str or ""
                    if answer.startswith('answer='):
                        answer = answer[7:].strip()
                    # Remove surrounding quotes
                    for quote in ['"', "'", '"""', "'''"]:
                        if answer.startswith(quote) and answer.endswith(quote):
                            answer = answer[len(quote):-len(quote)]
                            break
                    # Treat empty or missing answer as NO_COMPLETION_NEEDED
                    if not answer or answer.strip() == "":
                        answer = "NO_COMPLETION_NEEDED"
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
                
                # Convert types for calculate function (strict numeric acceptance)
                if tool_name == "calculate" and parsed_args:
                    if "num1" in parsed_args:
                        val1 = parsed_args["num1"]
                        parsed_args["num1"] = float(val1) if _is_strict_number(val1) else None
                    if "num2" in parsed_args:
                        val2 = parsed_args["num2"]
                        parsed_args["num2"] = float(val2) if _is_strict_number(val2) else None
                    if "duration" in parsed_args:
                        duration_val = parsed_args["duration"]
                        if duration_val and str(duration_val).lower() != 'none':
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
        
        # Get a value with flexible inputs
        value = await env.execute_tool(
            "search",
            {"metric": "net income", "ticker": "Apple", "period": "2023"}
        )
        print(f"Apple 2023 net income: {value}")

        value2 = await env.execute_tool(
            "search",
            {"metric": "price to earnings", "ticker": "alphabet", "period": "latest"}
        )
        print(f"Alphabet latest P/E: {value2}")
        
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
        test_response = 'search(metric="revenue", ticker="AAPL", period="2023FY")'
        parsed = parse_tool_calls_from_response(test_response)
        print(f"Parsed: {parsed}")
    
    asyncio.run(test_environment())