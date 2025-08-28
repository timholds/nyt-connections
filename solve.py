import json
import argparse
from typing import List, Dict, Any
from solvers import get_solver
from solvers.models import PuzzleSolution


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



def check_solution_correctness(predicted_solution: PuzzleSolution, ground_truth: Dict[str, Any]) -> bool:
    """
    Check if predicted solution matches ground truth.
    
    Args:
        predicted_solution: PuzzleSolution from model
        ground_truth: Ground truth from dataset
    
    Returns:
        True if correct, False otherwise
    """
    # Extract groups from both
    predicted_groups = [group.words for group in predicted_solution.groups]
    actual_groups = [group['words'] for group in ground_truth['groups']]
    
    # Normalize to sets of frozensets for comparison
    pred_normalized = {frozenset(group) for group in predicted_groups}
    actual_normalized = {frozenset(group) for group in actual_groups}
    
    return pred_normalized == actual_normalized


def main():
    """Main function to run the solver on test examples."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Solve NYT Connections puzzles")
    parser.add_argument("--file", type=str, default="examples_test.jsonl",
                        help="Path to JSONL file with puzzle examples (default: examples_test.jsonl)")
    parser.add_argument("--use-api", action="store_true",
                        help="Make actual API calls instead of dummy responses")
    parser.add_argument("--no-score", action="store_true",
                        help="Skip scoring the solution against ground truth")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                        help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--solver", type=str, default="baseline",
                        choices=["baseline", "few-shot", "dspy"],
                        help="Solver to use (default: baseline)")
    parser.add_argument("--limit", type=int, default=1,
                        help="Number of examples to process (default: 1)")
    args = parser.parse_args()
    
    # Load test examples
    examples = load_examples(args.file)
    print(f"Loaded {len(examples)} test examples from {args.file}")
    print(f"Using solver: {args.solver}")
    
    # Limit examples to process
    examples_to_process = examples[:args.limit]
    print(f"Processing {len(examples_to_process)} example(s) (--limit={args.limit})")
    
    # Get the solver instance once, passing all examples for few-shot/dspy solvers
    # They can use these examples for learning (excluding the current test example)
    solver = get_solver(args.solver, examples=examples)
    
    # Track overall statistics
    correct_count = 0
    total_count = 0
    
    # Process each example
    for idx, example in enumerate(examples_to_process):
        total_count += 1
        words = extract_words(example)
        print(f"\n{'='*60}")
        print(f"EXAMPLE {idx + 1}/{len(examples_to_process)}")
        print(f"{'='*60}")
        print(f"Processing example with words: {words}")
        
        # Solve the puzzle using the solver
        try:
            solution = solver.solve(words, use_api=args.use_api, model=args.model)
        except Exception as e:
            if "validation error" in str(e).lower() or "ValidationError" in str(type(e).__name__):
                print(f"\n❌ SOLVER FAILED: Could not generate valid solution after retries")
                print(f"Error: {str(e).split('For further')[0].strip()}")
                print("\nSkipping to next example...")
                continue
            else:
                raise  # Re-raise non-validation errors
        
        print("\n" + "="*50)
        print("PREDICTED SOLUTION:")
        print("-"*50)
        for i, group in enumerate(solution.groups, 1):
            print(f"{i}. {', '.join(group.words)}")
        print("\nREASONS:")
        for i, group in enumerate(solution.groups, 1):
            print(f"{i}. {group.reason}")
        
        print("\n" + "="*50)
        print("ACTUAL SOLUTION:")
        print("-"*50)
        actual_groups = example['solution']['groups']
        for i, group in enumerate(actual_groups, 1):
            print(f"{i}. {', '.join(group['words'])}")
        print("\nREASONS:")
        for i, group in enumerate(actual_groups, 1):
            print(f"{i}. {group['reason']}")
        
        # Check correctness by default (unless --no-score is passed)
        if not args.no_score:
            is_correct = check_solution_correctness(solution, example['solution'])
            print("\n" + "="*50)
            print(f"✓ Solution is {'CORRECT' if is_correct else 'INCORRECT'}")
            if is_correct:
                correct_count += 1
    
    # Print summary if multiple examples were processed
    if total_count > 1 and not args.no_score:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {correct_count}/{total_count} correct ({100*correct_count/total_count:.1f}%)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()