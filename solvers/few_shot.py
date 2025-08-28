from typing import List, Dict, Any, Optional
import json
from .base import BaseSolver


class FewShotSolver(BaseSolver):
    """Few-shot solver that includes examples in the prompt."""
    
    def __init__(self, examples_file: str = "examples_test.jsonl", num_examples: int = 3):
        """
        Initialize the few-shot solver.
        
        Args:
            examples_file: Path to JSONL file with examples
            num_examples: Number of examples to include in prompt
        """
        super().__init__()
        self.examples_file = examples_file
        self.num_examples = num_examples
        self.examples = self._load_examples()
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load examples from JSONL file."""
        examples = []
        try:
            with open(self.examples_file, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Examples file {self.examples_file} not found. Using no examples.")
        return examples
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format a single example for the prompt."""
        words = example["words"]
        solution = example["solution"]["groups"]
        
        formatted = f"Words: {', '.join(words)}\n\n"
        formatted += "Solution:\n"
        for i, group in enumerate(solution, 1):
            formatted += f"{i}. {', '.join(group['words'])}\n"
            formatted += f"   Reason: {group['reason']}\n"
        
        return formatted
    
    def get_system_prompt(self) -> str:
        """Get the system prompt with examples for the few-shot solver."""
        base_prompt = """You are an expert at solving NYT Connections puzzles. 
        Given 16 words, you need to group them into 4 groups of 4 words each.
        Each group should have a clear connecting theme or category.
        
        Consider various types of connections including:
        - Literal meanings and categories
        - Wordplay (palindromes, homophones, rhymes)
        - Words that can precede or follow a common word
        - Pop culture references
        - Common phrases or idioms
        
        Learn from the following examples to understand the patterns and reasoning:
        """
        
        # Add examples if available
        if self.examples and self.num_examples > 0:
            examples_to_use = self.examples[:min(self.num_examples, len(self.examples))]
            base_prompt += "\n\n" + "="*50 + "\n"
            for i, example in enumerate(examples_to_use, 1):
                base_prompt += f"\nExample {i}:\n"
                base_prompt += "-"*30 + "\n"
                base_prompt += self._format_example(example)
                if i < len(examples_to_use):
                    base_prompt += "\n"
            base_prompt += "="*50 + "\n\n"
            base_prompt += "Now apply similar reasoning to solve the new puzzle. Be concise but clear in your reasoning."
        
        return base_prompt
    
    def get_user_message(self, words: List[str]) -> str:
        """Format the user message for the few-shot solver."""
        return f"""Group these 16 words into 4 groups of 4 words each. 
    Each group should share a common theme or connection.
    
    Words: {', '.join(words)}
    
    Provide a clear, concise reason for each grouping, similar to the examples provided."""