"""
Agent wrapper for LLM-based autocomplete with multi-turn tool calling
Handles conversation management and interfaces with ART models
"""

from typing import List, Dict, Optional, Tuple, Any
import asyncio
import os
from environment import (
    FinancialEnvironment,
    parse_tool_calls_from_response,
    count_tool_calls,
    is_valid_single_tool_call_format,
    analyze_tool_calls,
)

SYSTEM_PROMPT = '''You are a financial data assistant. You will be given the start of a sentence in <input> tags, and you need to decide if continuing the sentence requires financial data.

If the continuation doesn't require financial data, use return_answer("").

If it does, use search(metric, company, period) e.g. search(revenue, Apple, latest) or search(capital_expenditures, NVDA, FY2023) to get the data needed. If you need to combine more than one piece of data, use calculate(num1, num2, operation) where operation can be "add", "subtract", "multiply", "divide".

After each tool use, you will receive the results and can call more tools. If a tool returns an invalid response, try again with different arguments.

When you have enough information, use return_answer() to continue the sentence e.g. return_answer("100 billion"). Do not include the <input> text in return_answer().'''

def _render_chatml(messages: List[Dict[str, str]]) -> str:
    """
    Render conversation into a ChatML-style transcript (similar to Ollama's default
    Qwen template), without tool schema injection. Primes an assistant turn.
    """
    parts: List[str] = []
    # First system message if present
    system_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), None)
    if system_msg:
        parts.append("<|im_start|>system")
        parts.append(system_msg)
        parts.append("<|im_end|>")

    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        content = m.get("content", "")
        if role == "user":
            parts.append("<|im_start|>user")
            parts.append(content)
            parts.append("<|im_end|>")
        elif role == "assistant":
            parts.append("<|im_start|>assistant")
            parts.append(content)
            parts.append("<|im_end|>")

    # Prime the next assistant turn
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)

class AutocompleteAgent:
    """
    Wrapper for LLM agents that perform financial autocomplete
    Manages multi-turn conversations and tool interactions
    """
    
    def __init__(self, model: Any = None, temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 128):
        """
        Initialize the autocomplete agent
        
        Args:
            model: ART model or OpenAI-compatible model
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
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
        max_turns: int = 7,
        max_tool_calls_per_turn: int = 1,
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
            {"role": "user", "content": f"<input>{text}</input>"}
        ]
        
        # Per-turn telemetry for rewards
        assistant_turn_lengths: List[int] = []
        assistant_turn_tool_calls_per_turn: List[int] = []
        assistant_turn_valid_format: List[bool] = []

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
            
            # Add assistant response to history FIRST so we always have a matching assistant turn
            # for any Choice/logprob object returned by the model (required by ART tokenization).
            self.conversation.append({"role": "assistant", "content": response_text})

            # Telemetry: compute per-turn stats based on raw assistant text
            resp_len = len(response_text or "")
            assistant_turn_lengths.append(resp_len)

            # Analyze tool calls (single source of truth for counts)
            analysis = analyze_tool_calls(response_text or "")
            tool_calls_count = analysis.get("count", 0)
            first_tool_call = analysis.get("first")
            assistant_turn_tool_calls_per_turn.append(tool_calls_count)
            # Valid format = exactly one tool call and nothing else
            is_valid_format = is_valid_single_tool_call_format(response_text or "")
            assistant_turn_valid_format.append(is_valid_format)

            if tool_calls_count == 0 or not first_tool_call:
                # No tool calls found: always prompt for tool usage (no direct completion fallback)
                self.conversation.append({
                    "role": "user",
                    "content": "Please use the available tools to complete the task."
                })
                continue
            if tool_calls_count > max_tool_calls_per_turn:
                # Reject turns with too many tool calls
                self.conversation.append({
                    "role": "user",
                    "content": f"Invalid: too many tool calls in one turn."
                })
                continue
            
            # Execute only the first tool call
            result = await self.environment.execute_tool(
                first_tool_call["tool"],
                first_tool_call.get("args", {})
            )
            
            # If we got a final answer, we're done
            if self.environment.is_complete():
                break
            
            # Add single tool result to conversation for next turn (trimmed format)
            self.conversation.append({
                "role": "user",
                "content": f"{result}"
            })
        
        # Prepare episode info
        episode_info = {
            "turns": len([m for m in self.conversation if m["role"] == "assistant"]),
            "tool_calls_count": len(self.environment.tool_calls),
            "completed": self.environment.is_complete(),
            "max_turns_reached": turn >= (max_turns - 1),
            # New telemetry for reward shaping
            "assistant_turn_lengths": assistant_turn_lengths,
            "assistant_turn_tool_calls_per_turn": assistant_turn_tool_calls_per_turn,
            "assistant_turn_valid_format": assistant_turn_valid_format,
            "invalid_format_turns_count": sum(1 for v in assistant_turn_valid_format if not v),
            "total_characters": sum(assistant_turn_lengths) if assistant_turn_lengths else 0,
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
                use_chatml = os.getenv("ART_USE_CHATML") == "1"
                if use_chatml:
                    prompt = _render_chatml(messages)
                    resp = await client.chat.completions.create(
                        model=self.model.name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stop=["<|im_end|>"],
                        logprobs=True,
                        store=False,
                    )
                else:
                    resp = await client.chat.completions.create(
                        model=self.model.name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
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
                top_p=self.top_p,
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