"""
Example of how to apply MIPRO optimizations to non-DSPy solvers.
This shows how you would modify FewShotSolver to use optimized components.
"""

from typing import List, Dict, Any, Optional
from solvers.few_shot import FewShotSolver
from apply_optimization import get_optimized_demos, get_optimized_prompts


class OptimizedFewShotSolver(FewShotSolver):
    """
    FewShotSolver enhanced with MIPRO-optimized examples and prompts.
    
    This demonstrates how to apply optimizations to a non-DSPy solver.
    """
    
    def __init__(self, model: str = "gpt-5-mini", num_examples: int = 3):
        # Load MIPRO-optimized examples instead of random ones
        optimized_demos = get_optimized_demos(model)
        
        # Convert DSPy format to FewShotSolver format
        examples = []
        for demo in optimized_demos[:num_examples]:
            # Parse the demo into the format FewShotSolver expects
            example = {
                "words": [w.strip() for w in demo.get("words", "").split(",")],
                "solution": {
                    "groups": []
                }
            }
            
            # Extract groups from the demo
            for i in range(1, 5):
                group_words = demo.get(f"group{i}_words", "")
                group_reason = demo.get(f"group{i}_reason", "")
                
                if group_words:
                    example["solution"]["groups"].append({
                        "words": [w.strip() for w in group_words.split(",")],
                        "reason": group_reason
                    })
            
            if len(example["solution"]["groups"]) == 4:
                examples.append(example)
        
        # Initialize with optimized examples
        super().__init__(examples=examples, num_examples=num_examples)
        self.model = model
        
        # Load optimized prompts
        self.optimized_prompts = get_optimized_prompts(model)
    
    def get_system_prompt(self, current_words: Optional[List[str]] = None) -> str:
        """Override to use MIPRO-optimized instructions if available."""
        
        # Check if we have an optimized instruction
        if self.optimized_prompts and 'reasoning' in self.optimized_prompts:
            # Use the optimized reasoning description as part of the prompt
            optimized_instruction = self.optimized_prompts['reasoning']
            
            base_prompt = f"""You are an expert at solving NYT Connections puzzles.
{optimized_instruction}

Given 16 words, group them into 4 groups of 4 words each.
Each group should have a clear connecting theme or category.

Here are some examples of solved puzzles:
"""
        else:
            # Fall back to original prompt
            base_prompt = super().get_system_prompt(current_words)
        
        # Add the optimized examples
        if self.examples:
            base_prompt += "\n\nExamples:\n"
            # Use only the MIPRO-selected best examples
            for i, example in enumerate(self.examples[:self.num_examples], 1):
                if current_words:
                    # Skip if this is the current puzzle
                    example_words = set(w.lower() for w in example['words'])
                    current_set = set(w.lower() for w in current_words)
                    if example_words == current_set:
                        continue
                
                base_prompt += f"\nExample {i}:\n"
                base_prompt += self._format_example(example)
                base_prompt += "\n---\n"
        
        return base_prompt


# Usage example
if __name__ == "__main__":
    import os
    
    # First, make sure you've run optimization:
    # python optimize_once.py --model gpt-5-mini
    
    # Then use the optimized solver
    solver = OptimizedFewShotSolver(model="gpt-5-mini")
    
    # Test words
    test_words = [
        "BANK", "RIVER", "STREAM", "CURRENT",
        "FLOW", "TIDE", "WAVE", "DRIFT",
        "CASH", "MONEY", "FUNDS", "CAPITAL",
        "LEAN", "TILT", "SLOPE", "ANGLE"
    ]
    
    # This will now use:
    # 1. The best 3 examples selected by MIPRO (not random)
    # 2. Optimized prompt instructions (if available)
    result = solver.solve(
        words=test_words,
        use_api=True,  # Set to True to actually call the API
        model="gpt-5-mini"
    )
    
    print("Optimized FewShotSolver Results:")
    for group in result.groups:
        print(f"- {', '.join(group.words)}: {group.reason}")