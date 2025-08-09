"""
Agent wrapper for LLM-based autocomplete with multi-turn tool calling
Handles conversation management and interfaces with ART models
"""

from typing import List, Dict, Optional, Tuple, Any
import asyncio
import os
from environment import FinancialEnvironment, parse_tool_calls_from_response

SYSTEM_PROMPT = '''You are a financial data assistant that completes partial text with accurate financial data.

Available tools (YOU MUST USE THESE):
- get_metrics(): returns list of available metrics
- get_tickers(): returns list of available tickers  
- get_value(metric, ticker, period): gets a specific value
  - Period format: "2024Q1" for Q1 2024, "2024FY" for fiscal year 2024, or "latest" for most recent
- calculate(num1, num2, operation, duration): operations are "add", "subtract", "multiply", "divide", "CAGR"
    - for CAGR, duration is the number of time periods
    - for CAGR, num1 is the final value and num2 is the initial value
- return_answer(answer): returns the final completion text (ONLY the completion, not the full text)

If the text doesn't need financial data, use: return_answer(answer="NO_COMPLETION_NEEDED")

IMPORTANT RULES:
1. You MUST use tools - do not provide commentary or ask questions, and do not make up values
2. This is a multi-turn conversation. After you call tools, you'll receive the results and can call more tools or return the answer
3. Use one tool call at a time. Do not nest tool calls.
4. Make sure to check the available metrics AND available tickers before calling get_value.
5. Always end with return_answer() to provide your completion
6. Only return the completion text, not the full sentence or any of the input text

EXAMPLE:
User: "The revenue for Apple in 2023 was"
You: get_metrics()
User: [list of metrics]
You: get_tickers()
User: [list of tickers]
You: get_value(metric="revenue", ticker="AAPL", period="2023FY")
User: [value]
You: return_answer(answer="$383.3 billion")'''

class AutocompleteAgent:
    """
    Wrapper for LLM agents that perform financial autocomplete
    Manages multi-turn conversations and tool interactions
    """
    
    def __init__(self, model: Any = None, temperature: float = 0.1, max_tokens: int = 500):
        """
        Initialize the autocomplete agent
        
        Args:
            model: ART model or OpenAI-compatible model
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation: List[Dict[str, str]] = []
        self.conversation_choices: List[dict] = []
        self.environment = FinancialEnvironment()
    
    def reset(self):
        """Reset the agent for a new episode"""
        self.conversation = []
        self.conversation_choices = []
        self.environment.reset()
    
    async def get_completion(
        self,
        text: str,
        max_turns: int = 10
    ) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get autocomplete for the given text using multi-turn tool interactions
        
        Args:
            text: The text to complete
            max_turns: Maximum number of conversation turns
            
        Returns:
            Tuple of (completion, tool_calls_log, episode_info)
        """
        self.reset()
        
        # Build initial conversation
        self.conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Complete this text with financial data, if needed. If not, return_answer(answer='NO_COMPLETION_NEEDED'): '{text}'"}
        ]
        
        # Multi-turn conversation loop
        for turn in range(max_turns):
            # Get model response
            response = await self._call_model(self.conversation)
            
            if response is None:
                break
            
            # Store response (both text and choice object if available)
            if isinstance(response, dict):
                response_text = response.get("content", "")
                self.conversation_choices.append(response)
            else:
                response_text = response
            
            # Check if no completion needed
            if "NO_COMPLETION_NEEDED" in response_text:
                await self.environment.execute_tool("return_answer", {"answer": "NO_COMPLETION_NEEDED"})
                break
            
            # Add assistant response to history
            self.conversation.append({"role": "assistant", "content": response_text})
            
            # Parse tool calls
            tool_calls = parse_tool_calls_from_response(response_text)
            
            if not tool_calls:
                # No tool calls found, prompt for tool usage
                if turn == 0 and len(response_text.strip()) > 0:
                    # First turn with no tools - treat as direct completion
                    await self.environment.execute_tool("return_answer", {"answer": response_text.strip()})
                    break
                else:
                    self.conversation.append({
                        "role": "user",
                        "content": "Please use the available tools to complete the task. Use return_answer(answer='your completion') to provide the final answer."
                    })
                    continue
            
            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                result = await self.environment.execute_tool(
                    tool_call["tool"],
                    tool_call.get("args", {})
                )
                
                # Check if this is the final answer
                if tool_call["tool"] == "return_answer":
                    break
                
                tool_results.append({
                    "tool": tool_call["tool"],
                    "args": tool_call.get("args", {}),
                    "result": result
                })
            
            # If we got a final answer, we're done
            if self.environment.is_complete():
                break
            
            # Add tool results to conversation for next turn
            if tool_results:
                tool_results_text = "Tool results:\n"
                for tr in tool_results:
                    tool_results_text += f"{tr['tool']}: {tr['result']}\n"
                self.conversation.append({"role": "user", "content": tool_results_text})
        
        # Prepare episode info
        episode_info = {
            "turns": len([m for m in self.conversation if m["role"] == "assistant"]),
            "tool_calls_count": len(self.environment.tool_calls),
            "completed": self.environment.is_complete(),
            "max_turns_reached": turn >= (max_turns - 1)
        }
        
        return (
            self.environment.get_final_answer(),
            self.environment.get_tool_calls_log(),
            episode_info
        )
    
    async def _call_model(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Call the underlying model
        
        Args:
            messages: Conversation messages
            
        Returns:
            Model response text or None if error
        """
        if self.model is None:
            # Default to OpenAI for testing
            return await self._call_openai(messages)
        
        # Check if this is an ART model
        if hasattr(self.model, 'inference_base_url'):
            # ART model
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                base_url=self.model.inference_base_url,
                api_key=self.model.inference_api_key,
            )
            
            try:
                resp = await client.chat.completions.create(
                    model=self.model.name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    logprobs=True,
                    store=False,
                )
                
                if resp and resp.choices:
                    # Return both content and choice for ART trajectory building
                    return {
                        "content": resp.choices[0].message.content.strip(),
                        "choice": resp.choices[0]
                    }
            except Exception as e:
                print(f"Error calling ART model: {e}")
                return None
        else:
            # Assume it's a callable or has a generate method
            if callable(self.model):
                return await self.model(messages)
            elif hasattr(self.model, 'generate'):
                return await self.model.generate(messages)
        
        return None
    
    async def _call_openai(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Fallback to OpenAI API for testing"""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None
    
    def get_conversation(self) -> List[Dict[str, str]]:
        """Get the full conversation history"""
        return self.conversation.copy()
    
    def get_messages_and_choices(self) -> List[dict]:
        """
        Get messages and choices for ART trajectory building
        Merges conversation messages with model choices
        """
        mc = []
        choice_i = 0
        
        for msg in self.conversation:
            if msg["role"] == "assistant":
                if choice_i < len(self.conversation_choices):
                    # Add the choice object if we have it
                    choice = self.conversation_choices[choice_i]
                    if isinstance(choice, dict) and "choice" in choice:
                        mc.append(choice["choice"])
                    else:
                        mc.append(msg)
                    choice_i += 1
                else:
                    mc.append(msg)
            else:
                mc.append(msg)
        
        return mc

# ============== Testing ==============

if __name__ == "__main__":
    async def test_agent():
        from database import setup_database
        
        # Setup database
        await setup_database()  # Requires TIINGO_API_KEY
        
        # Create agent (will use OpenAI for testing)
        agent = AutocompleteAgent()
        
        # Test completions
        test_cases = [
            "The revenue for Apple in 2023 was ",
            "Microsoft's net income in 2023Q4 was ",
            "The CFO mentioned that ",
        ]
        
        for text in test_cases:
            print(f"\nInput: {text}")
            
            completion, tool_calls, info = await agent.get_completion(text)
            
            print(f"Completion: {completion}")
            print(f"Episode info: {info}")
            print(f"Tool calls made: {len(tool_calls)}")
            
            if tool_calls:
                print("Tool sequence:")
                for tc in tool_calls:
                    print(f"  - {tc['tool']}({tc.get('arguments', {})})")
    
    asyncio.run(test_agent())