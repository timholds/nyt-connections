from abc import ABC, abstractmethod
from typing import List, Optional
import os
from datetime import datetime
from openai import OpenAI
from .models import GroupSolution, PuzzleSolution


class BaseSolver(ABC):
    """Base class for all Connections puzzle solvers."""
    
    # Model pricing as of 2024 (per 1M tokens)
    MODEL_PRICING = {
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},  # Doesn't support structured output
    }
    
    def __init__(self):
        """Initialize the solver."""
        self.client = None
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this solver."""
        pass
    
    @abstractmethod
    def get_user_message(self, words: List[str]) -> str:
        """Format the user message for this solver."""
        pass
    
    def log_api_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cost: float):
        """
        Log API cost to the tracking file.
        
        Args:
            model: Model name used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens  
            cost: Total cost in USD
        """
        log_file = "openai_api_costs.txt"
        
        # Read existing total if file exists
        existing_total = 0
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if line.startswith("TOTAL:"):
                        existing_total = float(line.split('$')[1].strip())
                        break
        
        # Calculate new total
        new_total = existing_total + cost
        
        # Append new entry
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | Model: {model} | ")
            f.write(f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion | ")
            f.write(f"Cost: ${cost:.4f}\n")
            f.write(f"TOTAL: ${new_total:.4f}\n")
            f.write("-" * 70 + "\n")
    
    def solve(self, words: List[str], use_api: bool = False, model: str = "gpt-4o-mini") -> PuzzleSolution:
        """
        Solve a Connections puzzle.
        
        Args:
            words: List of 16 words to group
            use_api: Whether to make actual API call (vs dummy response)
            model: OpenAI model to use
        
        Returns:
            PuzzleSolution with structured groups including reasoning
        """
        # Validate input
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        if len(set(words)) != 16:
            raise ValueError("All 16 words must be unique")
        
        # Get prompts from the specific solver implementation
        system_prompt = self.get_system_prompt()
        user_message = self.get_user_message(words)
        
        if use_api:
            if model not in self.MODEL_PRICING:
                raise ValueError(f"Unknown model: {model}. Add pricing to MODEL_PRICING dict.")
            
            # Initialize client if needed
            if not self.client:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Make actual API call with structured output (with retries for validation errors)
            max_retries = 3
            solution = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                    
                    # Add specific retry message based on the validation error
                    if attempt > 0 and last_error:
                        error_str = str(last_error)
                        retry_message = "Your previous response had a validation error. "
                        
                        # Customize message based on specific validation error
                        if "unique across all groups" in error_str:
                            retry_message += ("You included the same word in multiple groups. "
                                            "Remember: Each word must appear in EXACTLY ONE group. "
                                            "All 16 words must be used exactly once.")
                        elif "exactly 4 words" in error_str:
                            retry_message += ("One or more groups didn't have exactly 4 words. "
                                            "Remember: Each group must contain EXACTLY 4 words.")
                        elif "exactly 4 groups" in error_str:
                            retry_message += ("You didn't provide exactly 4 groups. "
                                            "Remember: You must create EXACTLY 4 groups.")
                        elif "Total must be 16" in error_str:
                            retry_message += ("The total number of words across all groups wasn't 16. "
                                            "Remember: Use all 16 words exactly once.")
                        else:
                            retry_message += f"Error details: {error_str}"
                        
                        retry_message += " Please try again."
                        messages.append({"role": "user", "content": retry_message})
                    
                    completion = self.client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=PuzzleSolution,
                    )
                    
                    # Extract the parsed solution
                    solution = completion.choices[0].message.parsed
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    # Check if it's a validation error and we have retries left
                    if "validation error" in str(e).lower() and attempt < max_retries - 1:
                        print(f"\nValidation error on attempt {attempt + 1}/{max_retries}: {str(e).split('For further')[0].strip()}")
                        print("Retrying with clarified instructions...")
                        continue
                    else:
                        raise  # Re-raise if it's a different error or we're out of retries
            
            # Track API costs
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            
            pricing = self.MODEL_PRICING[model]
            input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
            output_cost = (completion_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost
            
            # Log the cost
            self.log_api_cost(model, prompt_tokens, completion_tokens, total_cost)
            
            print(f"\nAPI Cost: ${total_cost:.4f} ({prompt_tokens} prompt + {completion_tokens} completion tokens)")
            
        else:
            # Print what would be sent to the LLM (dummy call for testing)
            print("=" * 80)
            print("WOULD SEND TO LLM:")
            print("-" * 40)
            print(f"System Prompt: {system_prompt[:200]}..." if len(system_prompt) > 200 else system_prompt)
            print("-" * 40)
            print(f"User Message: {user_message[:200]}..." if len(user_message) > 200 else user_message)
            print("=" * 80)
            
            # Create dummy response
            solution = PuzzleSolution(
                groups=[
                    GroupSolution(words=list(words[:4]), reason="dummy group 1"),
                    GroupSolution(words=list(words[4:8]), reason="dummy group 2"),
                    GroupSolution(words=list(words[8:12]), reason="dummy group 3"),
                    GroupSolution(words=list(words[12:16]), reason="dummy group 4")
                ]
            )
        
        return solution