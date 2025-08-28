import json
import os
import argparse
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


class GroupSolution(BaseModel):
    """A single group of 4 words with their connecting reason."""
    words: List[str] = Field(description="Exactly 4 words that belong together", min_length=4, max_length=4)
    reason: str = Field(description="The reason/theme connecting these 4 words")
    
    @field_validator('words')
    @classmethod
    def validate_words_count(cls, v):
        if len(v) != 4:
            raise ValueError(f"Each group must have exactly 4 words, got {len(v)}")
        return v


class PuzzleSolution(BaseModel):
    """Complete solution to a Connections puzzle."""
    groups: List[GroupSolution] = Field(
        description="Exactly 4 groups of 4 words each",
        min_length=4,
        max_length=4
    )
    
    @field_validator('groups')
    @classmethod
    def validate_groups(cls, v):
        if len(v) != 4:
            raise ValueError(f"Must have exactly 4 groups, got {len(v)}")
        
        # Check that all words are unique across groups
        all_words = []
        for group in v:
            all_words.extend(group.words)
        
        if len(all_words) != len(set(all_words)):
            raise ValueError("Words must be unique across all groups")
        
        if len(all_words) != 16:
            raise ValueError(f"Total must be 16 unique words, got {len(set(all_words))}")
        
        return v


def load_examples(filepath: str = "examples_test.jsonl") -> List[Dict[str, Any]]:
    """Load full examples from JSONL file including words and solutions."""
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def extract_words(example: Dict[str, Any]) -> List[str]:
    """Extract just the words from an example."""
    return example["words"]

def solve_puzzle(words: List[str], system_prompt: str = "", use_api: bool = False) -> PuzzleSolution:
    """
    Solve a Connections puzzle using OpenAI API with structured output.
    
    Args:
        words: List of 16 words to group
        system_prompt: System prompt for the model
        use_api: Whether to make actual API call (vs dummy response)
    
    Returns:
        PuzzleSolution with structured groups including reasoning
    """
    if not system_prompt:
        system_prompt = """You are an expert at solving NYT Connections puzzles. 
        Given 16 words, you need to group them into 4 groups of 4 words each.
        Each group should have a clear connecting theme or category.
        Consider various types of connections including:
        - Literal meanings and categories
        - Wordplay (palindromes, homophones, rhymes)
        - Words that can precede or follow a common word
        - Pop culture references
        - Common phrases or idioms
        Be concise but clear in your reasoning."""
    
    # Validate input
    if len(words) != 16:
        raise ValueError(f"Expected 16 words, got {len(words)}")
    
    if len(set(words)) != 16:
        raise ValueError("All 16 words must be unique")
    
    # Format the user message with the puzzle
    user_message = f"""Group these 16 words into 4 groups of 4 words each. 
    Each group should share a common theme or connection.
    
    Words: {', '.join(words)}
    
    Provide a clear, concise reason for each grouping."""
    
    if use_api:
        # Make actual API call with structured output
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Cost-effective model that supports structured output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format=PuzzleSolution,
        )
        
        # Extract the parsed solution - guaranteed to match our schema
        solution = completion.choices[0].message.parsed
        
    else:
        # Print what would be sent to the LLM (dummy call for now)
        print("=" * 80)
        print("WOULD SEND TO LLM:")
        print("-" * 40)
        print(f"System Prompt: {system_prompt[:200]}..." if len(system_prompt) > 200 else system_prompt)
        print("-" * 40)
        print(f"User Message: {user_message[:200]}..." if len(user_message) > 200 else user_message)
        print("=" * 80)
        
        # Create dummy response using Pydantic model
        solution = PuzzleSolution(
            groups=[
                GroupSolution(words=list(words[:4]), reason="dummy group 1"),
                GroupSolution(words=list(words[4:8]), reason="dummy group 2"),
                GroupSolution(words=list(words[8:12]), reason="dummy group 3"),
                GroupSolution(words=list(words[12:16]), reason="dummy group 4")
            ]
        )
    
    return solution

def main():
    """Main function to run the solver on test examples."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Solve NYT Connections puzzles")
    parser.add_argument("--file", type=str, default="examples_test.jsonl",
                        help="Path to JSONL file with puzzle examples (default: examples_test.jsonl)")
    parser.add_argument("--use-api", action="store_true",
                        help="Make actual API calls instead of dummy responses")
    args = parser.parse_args()
    
    # Load test examples
    examples = load_examples(args.file)
    print(f"Loaded {len(examples)} test examples from {args.file}")
    
    # Process first example as a demo
    if examples:
        first_example = examples[0]
        words = extract_words(first_example)
        print(f"\nProcessing first example with words: {words}")
        
        # Solve the puzzle (only passes words, not the solution)
        solution = solve_puzzle(words, use_api=args.use_api)
        
        print("\nSolution returned (Pydantic validated):")
        print(solution.model_dump_json(indent=2))
        
        print("\nActual solution from dataset:")
        print(json.dumps(first_example['solution'], indent=2))

if __name__ == "__main__":
    main()