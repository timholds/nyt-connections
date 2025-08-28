import json
import argparse
from typing import List, Dict, Any
from solvers import get_solver
from solvers.models import PuzzleSolution
from score import check_solution, calculate_detailed_metrics


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



def check_solution_with_metrics(predicted_solution: PuzzleSolution, ground_truth: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    """
    Check if predicted solution matches ground truth and get detailed metrics.
    
    Args:
        predicted_solution: PuzzleSolution from model
        ground_truth: Ground truth from dataset
    
    Returns:
        Tuple of (is_correct, metrics)
    """
    # Extract groups from both
    predicted_groups = [group.words for group in predicted_solution.groups]
    actual_groups = [group['words'] for group in ground_truth['groups']]
    
    # Check if correct using imported function
    is_correct = check_solution(predicted_groups, actual_groups)
    
    # Get detailed metrics
    metrics = calculate_detailed_metrics(predicted_groups, actual_groups)
    
    return is_correct, metrics


def run_single_solver(solver_name: str, examples: List[Dict[str, Any]], examples_to_process: List[Dict[str, Any]], 
                     use_api: bool, model: str, no_score: bool, verbose: bool = True) -> Dict[str, Any]:
    """Run a single solver on the examples and return results."""
    # Get the solver instance
    solver = get_solver(solver_name, examples=examples)
    
    if verbose:
        print(f"\nUsing solver: {solver_name}")
    
    # Track statistics for this solver
    correct_count = 0
    total_count = 0
    total_group_accuracy = 0
    
    # Process each example
    for idx, example in enumerate(examples_to_process):
        total_count += 1
        words = extract_words(example)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"EXAMPLE {idx + 1}/{len(examples_to_process)}")
            print(f"{'='*60}")
            print(f"Processing example with words: {words}")
        
        # Solve the puzzle using the solver
        try:
            solution = solver.solve(words, use_api=use_api, model=model)
            
            # Validate the solution structure
            if not solution or not solution.groups or len(solution.groups) != 4:
                if verbose:
                    print(f"\n⚠️ WARNING: Solver returned invalid structure")
                    print(f"Groups returned: {len(solution.groups) if solution and solution.groups else 0}")
                continue
                
            # Check each group has exactly 4 words
            for i, group in enumerate(solution.groups):
                if len(group.words) != 4:
                    if verbose:
                        print(f"\n⚠️ WARNING: Group {i+1} has {len(group.words)} words instead of 4")
                    
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Handle various error types
            if "validation error" in error_msg.lower() or "ValidationError" in error_type:
                if verbose:
                    print(f"\n❌ VALIDATION ERROR: Solver couldn't produce valid groups after retries")
                    if "For further" in error_msg:
                        error_msg = error_msg.split('For further')[0].strip()
                    print(f"Details: {error_msg}")
                    print("Skipping to next example...")
                continue
                
            elif "api" in error_msg.lower() or "openai" in error_msg.lower():
                if verbose:
                    print(f"\n❌ API ERROR: {error_type}")
                    print(f"Details: {error_msg}")
                    print("Skipping to next example...")
                continue
                
            elif "Missing embeddings" in error_msg:
                if verbose:
                    print(f"\n❌ EMBEDDING ERROR: {error_msg}")
                    print("Consider regenerating embeddings or using a different solver")
                    print("Skipping to next example...")
                continue
                
            else:
                if verbose:
                    print(f"\n❌ UNEXPECTED ERROR in {solver_name} solver")
                raise
        
        if verbose:
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
        
        # Check correctness
        if not no_score:
            is_correct, metrics = check_solution_with_metrics(solution, example['solution'])
            if verbose:
                print("\n" + "="*50)
                print(f"✓ Solution is {'CORRECT' if is_correct else 'INCORRECT'}")
                print(f"  Groups correct: {metrics['num_correct_groups']}/4 ({metrics['group_accuracy']*100:.0f}%)")
            if is_correct:
                correct_count += 1
            total_group_accuracy += metrics['group_accuracy']
    
    # Calculate final metrics
    total_accuracy = 100 * correct_count / total_count if total_count > 0 else 0
    avg_group_accuracy = 100 * total_group_accuracy / total_count if total_count > 0 else 0
    
    return {
        'solver': solver_name,
        'total_accuracy': total_accuracy,
        'group_accuracy': avg_group_accuracy,
        'correct': correct_count,
        'total': total_count
    }


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
                        choices=["baseline", "few-shot", "dspy", "multi-stage", "hungarian", "constraint"],
                        help="Solver to use (default: baseline)")
    parser.add_argument("--all", action="store_true",
                        help="Test all available solvers and show comparison")
    parser.add_argument("--limit", type=int, default=1,
                        help="Number of examples to process (default: 1)")
    args = parser.parse_args()
    
    # Load test examples
    examples = load_examples(args.file)
    print(f"Loaded {len(examples)} test examples from {args.file}")
    
    # Limit examples to process
    examples_to_process = examples[:args.limit]
    print(f"Processing {len(examples_to_process)} example(s) (--limit={args.limit})")
    
    if args.all:
        # Run all solvers
        all_solvers = ["baseline", "few-shot", "dspy", "multi-stage", "hungarian", "constraint"]
        results = []
        
        print("\n" + "="*70)
        print("TESTING ALL SOLVERS")
        print("="*70)
        
        for solver_name in all_solvers:
            print(f"\n{'='*70}")
            print(f"TESTING SOLVER: {solver_name}")
            print(f"{'='*70}")
            
            result = run_single_solver(
                solver_name, examples, examples_to_process,
                args.use_api, args.model, args.no_score, verbose=False
            )
            results.append(result)
            
            # Print brief summary for this solver
            print(f"\n✓ {solver_name} completed:")
            print(f"  Total Accuracy: {result['total_accuracy']:.1f}% ({result['correct']}/{result['total']})")
            print(f"  Group Accuracy: {result['group_accuracy']:.1f}%")
        
        # Print comparison table
        print("\n" + "="*70)
        print("SOLVER COMPARISON")
        print("="*70)
        print(f"{'Solver':<15} {'Total Acc':<12} {'Group Acc':<12} {'Correct':<10} {'Total':<10}")
        print("-"*70)
        
        # Sort results by total accuracy
        results.sort(key=lambda x: x['total_accuracy'], reverse=True)
        
        for result in results:
            print(f"{result['solver']:<15} {result['total_accuracy']:>6.1f}%      {result['group_accuracy']:>6.1f}%      "
                  f"{result['correct']:>6}/{result['total']:<3}     {result['total']:<10}")
        
        # Save all results to experiment history
        if args.use_api and len(examples_to_process) > 0:
            import csv
            import os
            from datetime import datetime
            
            history_file = "experiments.csv"
            file_exists = os.path.exists(history_file)
            
            with open(history_file, 'a', newline='') as f:
                fieldnames = ['timestamp', 'solver', 'model', 'total_acc', 'group_acc', 
                             'correct', 'total', 'test_file', 'mode']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                for result in results:
                    writer.writerow({
                        'timestamp': datetime.now().isoformat(),
                        'solver': result['solver'],
                        'model': args.model,
                        'total_acc': result['total_accuracy'],
                        'group_acc': result['group_accuracy'],
                        'correct': result['correct'],
                        'total': result['total'],
                        'test_file': os.path.basename(args.file),
                        'mode': 'api' if args.use_api else 'dummy'
                    })
            
            print(f"\n📊 All experiments logged to: {history_file}")
            print(f"   Run 'python score.py --compare' to see full experiment history")
            
    else:
        # Run single solver (original behavior)
        result = run_single_solver(
            args.solver, examples, examples_to_process,
            args.use_api, args.model, args.no_score, verbose=True
        )
        
        # Print summary if multiple examples were processed
        if result['total'] > 1 and not args.no_score:
            print(f"\n{'='*60}")
            print(f"SUMMARY:")
            print(f"  Total Accuracy: {result['total_accuracy']:.1f}% ({result['correct']}/{result['total']} puzzles)")
            print(f"  Group Accuracy: {result['group_accuracy']:.1f}%")
            print(f"{'='*60}")
            
            # Save to experiment history if running multiple examples
            if args.use_api:  # Only track real API experiments
                import csv
                import os
                from datetime import datetime
                
                history_file = "experiments.csv"
                file_exists = os.path.exists(history_file)
                
                with open(history_file, 'a', newline='') as f:
                    fieldnames = ['timestamp', 'solver', 'model', 'total_acc', 'group_acc', 
                                 'correct', 'total', 'test_file', 'mode']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerow({
                        'timestamp': datetime.now().isoformat(),
                        'solver': args.solver,
                        'model': args.model,
                        'total_acc': result['total_accuracy'],
                        'group_acc': result['group_accuracy'],
                        'correct': result['correct'],
                        'total': result['total'],
                        'test_file': os.path.basename(args.file),
                        'mode': 'api' if args.use_api else 'dummy'
                    })
                
                print(f"\n📊 Experiment logged to: {history_file}")
                print(f"   Run 'python score.py --compare' to see experiment history")


if __name__ == "__main__":
    main()