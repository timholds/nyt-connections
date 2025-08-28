import json
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from solve import solve_puzzle, PuzzleSolution


def normalize_groups(groups: List[List[str]]) -> Set[frozenset]:
    """
    Convert groups to a set of frozensets for order-independent comparison.
    Each group becomes a frozenset (order within group doesn't matter).
    The collection of groups becomes a set (order of groups doesn't matter).
    """
    return {frozenset(group) for group in groups}


def check_solution(predicted: List[List[str]], actual: List[List[str]]) -> bool:
    """
    Check if predicted solution matches actual solution.
    Order of groups and order within groups don't matter.
    
    Args:
        predicted: List of 4 groups of 4 words each (from model)
        actual: List of 4 groups of 4 words each (ground truth)
    
    Returns:
        True if solutions match, False otherwise
    """
    # Normalize both to sets of frozensets
    pred_normalized = normalize_groups(predicted)
    actual_normalized = normalize_groups(actual)
    
    # Check if they're identical
    return pred_normalized == actual_normalized


def calculate_detailed_metrics(predicted: List[List[str]], actual: List[List[str]]) -> Dict[str, Any]:
    """Calculate comprehensive validation metrics for a single puzzle."""
    metrics = {}
    
    pred_sets = [set(g) for g in predicted]
    actual_sets = [set(g) for g in actual]
    
    # 1. Exact Match
    metrics['exact_match'] = int(normalize_groups(predicted) == normalize_groups(actual))
    
    # 2. Group Accuracy
    correct_groups = sum(1 for p in pred_sets if p in actual_sets)
    metrics['group_accuracy'] = correct_groups / 4
    metrics['num_correct_groups'] = correct_groups
    
    # 3. Word Placement Accuracy
    word_to_actual = {word: i for i, group in enumerate(actual) for word in group}
    word_to_pred = {word: i for i, group in enumerate(predicted) for word in group}
    
    # Find correct word placements by checking if word's group matches any actual group
    correct_words = 0
    for word in word_to_actual:
        if word in word_to_pred:
            pred_group = set(predicted[word_to_pred[word]])
            if pred_group in actual_sets:
                actual_idx = next(i for i, g in enumerate(actual_sets) if g == pred_group)
                if word in actual[actual_idx]:
                    correct_words += 1
    metrics['word_accuracy'] = correct_words / 16
    
    # 4. Pairwise Accuracy
    correct_pairs = 0
    total_pairs = 0
    all_words = list(word_to_actual.keys())
    for i, w1 in enumerate(all_words):
        for w2 in all_words[i+1:]:
            total_pairs += 1
            actual_together = word_to_actual[w1] == word_to_actual[w2]
            pred_together = word_to_pred.get(w1, -1) == word_to_pred.get(w2, -2)
            if actual_together == pred_together:
                correct_pairs += 1
    metrics['pairwise_accuracy'] = correct_pairs / total_pairs if total_pairs > 0 else 0
    
    # 5. Two-word swap detection
    metrics['is_two_word_swap'] = 0
    if correct_groups == 2:
        wrong_pred = [p for p in pred_sets if p not in actual_sets]
        if len(wrong_pred) == 2:
            combined = wrong_pred[0] | wrong_pred[1]
            matching_actual = [a for a in actual_sets if len(a & combined) > 0]
            if len(matching_actual) == 2 and combined == (matching_actual[0] | matching_actual[1]):
                # Check if it's exactly 2 words swapped
                diff1 = wrong_pred[0] - matching_actual[0]
                diff2 = wrong_pred[1] - matching_actual[1]
                if len(diff1) <= 2 and len(diff2) <= 2:
                    metrics['is_two_word_swap'] = 1
    
    # 6. Partial overlap per group
    overlaps = []
    for pred_group in pred_sets:
        best_overlap = max((len(pred_group & actual_group) for actual_group in actual_sets), default=0)
        overlaps.append(best_overlap)
    metrics['avg_group_overlap'] = sum(overlaps) / (4 * 4)  # normalized by max possible (4)
    
    return metrics


def score_single_puzzle(words: List[str], ground_truth: List[List[str]], use_api: bool = True, verbose: bool = False) -> Tuple[bool, PuzzleSolution, Dict[str, Any]]:
    """
    Score a single puzzle by comparing model output to ground truth.
    
    Args:
        words: List of 16 words to group
        ground_truth: The correct grouping
        use_api: Whether to use actual API or dummy response
        verbose: Whether to print detailed output
    
    Returns:
        Tuple of (is_correct, solution)
    """
    # Get model's solution
    solution = solve_puzzle(words, use_api=use_api)
    
    # Extract just the word groups from structured output
    predicted_groups = [group.words for group in solution.groups]
    
    # Check if correct
    is_correct = check_solution(predicted_groups, ground_truth)
    
    if verbose:
        print(f"\nWords: {words}")
        print(f"Predicted: {predicted_groups}")
        print(f"Actual: {ground_truth}")
        print(f"Correct: {is_correct}")
        if not is_correct:
            # Show which groups matched and which didn't
            pred_normalized = normalize_groups(predicted_groups)
            actual_normalized = normalize_groups(ground_truth)
            matches = pred_normalized & actual_normalized
            print(f"Matching groups: {len(matches)}/4")
    
    return is_correct, solution


def score_dataset(filepath: str, use_api: bool = True, limit: int = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Score model performance on entire dataset.
    
    Args:
        filepath: Path to JSONL file with examples
        use_api: Whether to use actual API
        limit: Limit number of examples to process (None for all)
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with scoring results
    """
    # Load examples
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    if limit:
        examples = examples[:limit]
    
    # Score each example
    results = []
    correct_count = 0
    
    for i, example in enumerate(examples):
        words = example["words"]
        ground_truth = example["solution"]["groups"]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Example {i+1}/{len(examples)}")
        
        is_correct, solution = score_single_puzzle(words, ground_truth, use_api=use_api, verbose=verbose)
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "example_id": i,
            "correct": is_correct,
            "predicted": [group.model_dump() for group in solution.groups],
            "actual": ground_truth
        })
        
        # Print running accuracy
        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            accuracy = correct_count / (i + 1) * 100
            print(f"\nProgress: {i+1}/{len(examples)} - Accuracy: {accuracy:.1f}% ({correct_count}/{i+1} correct)")
    
    # Calculate final metrics
    accuracy = correct_count / len(examples) * 100 if examples else 0
    
    return {
        "total_examples": len(examples),
        "correct": correct_count,
        "incorrect": len(examples) - correct_count,
        "accuracy": accuracy,
        "results": results
    }


def main():
    """Main function to run scoring on test set."""
    parser = argparse.ArgumentParser(description="Score NYT Connections puzzle solver")
    parser.add_argument("test_file", type=str, help="Path to test JSONL file")
    parser.add_argument("--use-api", action="store_true", help="Use actual API calls (default: dummy mode)")
    parser.add_argument("--limit", type=int, help="Limit number of examples to score")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output for each example")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--output", type=str, help="Custom output file path (default: results_<timestamp>.json)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.test_file):
        print(f"Error: File {args.test_file} not found")
        return
    
    print("="*60)
    print("NYT CONNECTIONS PUZZLE SOLVER - SELF-GRADING")
    print("="*60)
    print(f"Test file: {args.test_file}")
    if args.limit:
        print(f"Limiting to first {args.limit} examples")
    print(f"Mode: {'API' if args.use_api else 'DUMMY (no API calls)'}")
    print("="*60)
    
    # Run scoring
    results = score_dataset(
        args.test_file, 
        use_api=args.use_api,
        limit=args.limit,
        verbose=args.verbose
    )
    
    # Print summary with better formatting
    print("\n" + "="*60)
    print("SELF-GRADING RESULTS")
    print("="*60)
    print(f"Total puzzles tested: {results['total_examples']}")
    print(f"Puzzles solved correctly: {results['correct']}")
    print(f"Puzzles solved incorrectly: {results['incorrect']}")
    print("-"*60)
    print(f"ACCURACY: {results['accuracy']:.2f}%")
    print("="*60)
    
    # Save results automatically unless --no-save
    if not args.no_save:
        # Generate filename with timestamp if not specified
        if args.output:
            output_file = args.output
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results_{timestamp}.json"
        
        # Add metadata to results
        results["metadata"] = {
            "test_file": args.test_file,
            "timestamp": datetime.now().isoformat(),
            "mode": "api" if args.use_api else "dummy",
            "limit": args.limit
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
    
    # Return exit code based on success (useful for CI/CD)
    return 0 if results['accuracy'] > 0 else 1


if __name__ == "__main__":
    main()