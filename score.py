import json
import argparse
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple, Optional
from solvers import get_solver
from solvers.models import PuzzleSolution
import wandb


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


def score_single_puzzle(words: List[str], ground_truth: List[List[str]], solver_name: str = "baseline", model: str = "gpt-4o-mini", use_api: bool = True, verbose: bool = False, log_to_wandb: bool = False) -> Tuple[bool, PuzzleSolution, Dict[str, Any]]:
    """
    Score a single puzzle by comparing model output to ground truth.
    
    Args:
        words: List of 16 words to group
        ground_truth: The correct grouping
        solver_name: Name of solver to use (baseline, few-shot, dspy)
        model: Model name to use (gpt-4o-mini, gpt-4o, etc.)
        use_api: Whether to use actual API or dummy response
        verbose: Whether to print detailed output
        log_to_wandb: Whether to log metrics to W&B
    
    Returns:
        Tuple of (is_correct, solution, metrics)
    """
    # Get solver and solve puzzle
    solver = get_solver(solver_name)
    solution = solver.solve(words, use_api=use_api, model=model)
    
    # Extract just the word groups from structured output
    predicted_groups = [group.words for group in solution.groups]
    
    # Check if correct
    is_correct = check_solution(predicted_groups, ground_truth)
    
    # Calculate detailed metrics
    metrics = calculate_detailed_metrics(predicted_groups, ground_truth)
    
    # Log to W&B if enabled
    if log_to_wandb and wandb.run is not None:
        wandb.log({
            "puzzle/exact_match": metrics['exact_match'],
            "puzzle/group_accuracy": metrics['group_accuracy'],
            "puzzle/word_accuracy": metrics['word_accuracy'],
            "puzzle/pairwise_accuracy": metrics['pairwise_accuracy'],
            "puzzle/is_two_word_swap": metrics['is_two_word_swap'],
            "puzzle/avg_group_overlap": metrics['avg_group_overlap'],
            "puzzle/num_correct_groups": metrics['num_correct_groups'],
        })
    
    if verbose:
        print(f"\nWords: {words}")
        print(f"Predicted: {predicted_groups}")
        print(f"Actual: {ground_truth}")
        print(f"Correct: {is_correct}")
        print(f"\nDetailed Metrics:")
        print(f"  Correct Groups: {metrics['num_correct_groups']}/4")
        print(f"  Word Accuracy: {metrics['word_accuracy']*100:.1f}%")
        print(f"  Pairwise Accuracy: {metrics['pairwise_accuracy']*100:.1f}%")
        print(f"  Avg Group Overlap: {metrics['avg_group_overlap']*100:.1f}%")
        if metrics['is_two_word_swap']:
            print(f"  ⚠️  Two-word swap detected!")
    
    return is_correct, solution, metrics


def score_dataset(filepath: str, solver_name: str = "baseline", model: str = "gpt-4o-mini", use_api: bool = True, limit: int = None, verbose: bool = False, 
                 wandb_config: Optional[Dict[str, Any]] = None, experiment_name: Optional[str] = None) -> Dict[str, Any]:
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
    
    # Initialize W&B if config provided
    if wandb_config:
        run_name = experiment_name or f"solver_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="nyt-connections",
            name=run_name,
            config=wandb_config,
            tags=["scoring", "solver"]
        )
        wandb.config.update({
            "test_file": filepath,
            "use_api": use_api,
            "limit": limit,
            "num_examples": len(examples)
        })
    
    # Score each example
    results = []
    correct_count = 0
    
    # Aggregate metrics
    all_metrics = {
        'exact_match': [],
        'group_accuracy': [],
        'word_accuracy': [],
        'pairwise_accuracy': [],
        'is_two_word_swap': [],
        'avg_group_overlap': [],
        'num_correct_groups': []
    }
    
    for i, example in enumerate(examples):
        words = example["words"]
        ground_truth = example["solution"]["groups"]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Example {i+1}/{len(examples)}")
        
        is_correct, solution, metrics = score_single_puzzle(
            words, ground_truth,
            solver_name=solver_name,
            model=model,
            use_api=use_api, 
            verbose=verbose,
            log_to_wandb=(wandb_config is not None)
        )
        
        # Aggregate metrics
        for key, value in metrics.items():
            if key in all_metrics:
                all_metrics[key].append(value)
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "example_id": i,
            "correct": is_correct,
            "predicted": [group.model_dump() for group in solution.groups],
            "actual": ground_truth,
            "metrics": metrics
        })
        
        # Print running accuracy
        if (i + 1) % 10 == 0 or (i + 1) == len(examples):
            accuracy = correct_count / (i + 1) * 100
            print(f"\nProgress: {i+1}/{len(examples)} - Accuracy: {accuracy:.1f}% ({correct_count}/{i+1} correct)")
    
    # Calculate final metrics
    accuracy = correct_count / len(examples) * 100 if examples else 0
    
    # Calculate aggregate statistics
    summary_metrics = {}
    for key, values in all_metrics.items():
        if values:
            summary_metrics[f"mean_{key}"] = sum(values) / len(values)
            summary_metrics[f"min_{key}"] = min(values)
            summary_metrics[f"max_{key}"] = max(values)
    
    # Log final metrics to W&B
    if wandb_config and wandb.run is not None:
        wandb.summary["final_accuracy"] = accuracy
        wandb.summary["total_correct"] = correct_count
        wandb.summary["total_incorrect"] = len(examples) - correct_count
        
        # Log aggregate metrics
        for key, value in summary_metrics.items():
            wandb.summary[key] = value
        
        # Create a table with detailed results
        table = wandb.Table(columns=["example_id", "correct", "exact_match", "group_accuracy", 
                                     "word_accuracy", "pairwise_accuracy", "is_two_word_swap"])
        for result in results:
            table.add_data(
                result["example_id"],
                result["correct"],
                result["metrics"]["exact_match"],
                result["metrics"]["group_accuracy"],
                result["metrics"]["word_accuracy"],
                result["metrics"]["pairwise_accuracy"],
                result["metrics"]["is_two_word_swap"]
            )
        wandb.log({"results_table": table})
        
        wandb.finish()
    
    return {
        "total_examples": len(examples),
        "correct": correct_count,
        "incorrect": len(examples) - correct_count,
        "accuracy": accuracy,
        "summary_metrics": summary_metrics,
        "results": results
    }


def show_experiment_history(n: int = 10) -> None:
    """Display recent experiment results in a clean table."""
    history_file = "experiments.csv"
    if not os.path.exists(history_file):
        print("No experiment history found. Run some experiments first!")
        return
    
    with open(history_file, 'r') as f:
        reader = csv.DictReader(f)
        experiments = list(reader)
    
    if not experiments:
        return
    
    # Take last n experiments
    recent = experiments[-n:] if len(experiments) > n else experiments
    
    print("\n" + "="*85)
    print(f"📊 EXPERIMENT HISTORY (last {len(recent)} runs)")
    print("="*85)
    print(f"{'Date':<12} {'Solver':<10} {'Model':<12} {'Total%':<9} {'Group%':<9} {'Correct':<10}")
    print("-"*85)
    
    for exp in recent:
        date = datetime.fromisoformat(exp['timestamp']).strftime("%m/%d %H:%M")
        print(f"{date:<12} {exp['solver']:<10} {exp['model']:<12} "
              f"{float(exp['total_acc']):<8.1f}% {float(exp['group_acc']):<8.1f}% "
              f"{exp['correct']}/{exp['total']:<8}")
    print("="*85)


def main():
    """Main function to run scoring on test set."""
    parser = argparse.ArgumentParser(description="Score NYT Connections puzzle solver")
    parser.add_argument("test_file", type=str, nargs='?', default="examples_test.jsonl", 
                       help="Path to test JSONL file (default: examples_test.jsonl)")
    parser.add_argument("--compare", action="store_true", help="Show experiment history and exit")
    parser.add_argument("--use-api", action="store_true", help="Use actual API calls (default: dummy mode)")
    parser.add_argument("--limit", type=int, help="Limit number of examples to score")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output for each example")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--output", type=str, help="Custom output file path (default: results_<timestamp>.json)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--experiment-name", type=str, help="Name for the W&B experiment")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                       choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                       help="OpenAI model to use")
    parser.add_argument("--solver", type=str, default="baseline",
                       choices=["baseline", "few-shot", "dspy"],
                       help="Solver to use (baseline, few-shot, or dspy)")
    
    args = parser.parse_args()
    
    # If --compare flag, show history and exit
    if args.compare:
        show_experiment_history()
        return 0
    
    # Check if file exists
    if not os.path.exists(args.test_file):
        print(f"Error: File {args.test_file} not found")
        return
    
    print("="*60)
    print("NYT CONNECTIONS PUZZLE SOLVER - SELF-GRADING")
    print("="*60)
    print(f"Test file: {args.test_file}")
    print(f"Solver: {args.solver}")
    print(f"Model: {args.model}")
    if args.limit:
        print(f"Limiting to first {args.limit} examples")
    print(f"Mode: {'API' if args.use_api else 'DUMMY (no API calls)'}")
    print("="*60)
    
    # Prepare W&B config if requested
    wandb_config = None
    if not args.no_wandb:
        wandb_config = {
            "model": args.model,
            "solver": args.solver,
            "test_file": os.path.basename(args.test_file),
            "use_api": args.use_api,
            "limit": args.limit,
        }
    
    # Run scoring
    results = score_dataset(
        args.test_file,
        solver_name=args.solver,
        model=args.model,
        use_api=args.use_api,
        limit=args.limit,
        verbose=args.verbose,
        wandb_config=wandb_config,
        experiment_name=args.experiment_name
    )
    
    # Print compact summary with key metrics highlighted
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)
    
    # Key metrics prominently displayed
    group_acc = results['summary_metrics'].get('mean_group_accuracy', 0) * 100 if 'summary_metrics' in results else 0
    print(f"📊 SOLVER: {args.solver.upper()}  |  MODEL: {args.model}")
    print(f"📈 TOTAL ACCURACY: {results['accuracy']:.1f}% ({results['correct']}/{results['total_examples']})")
    print(f"🎯 GROUP ACCURACY: {group_acc:.1f}%")
    print("-"*70)
    
    # Compact detailed metrics in table format
    if 'summary_metrics' in results and results['summary_metrics']:
        print("DETAILED METRICS:")
        print(f"{'Metric':<20} {'Mean':<8} {'Min':<8} {'Max':<8}")
        print("-" * 46)
        
        metric_display = {
            'exact_match': 'Perfect Solutions',
            'group_accuracy': 'Group Accuracy', 
            'word_accuracy': 'Word Accuracy',
            'pairwise_accuracy': 'Pairwise Accuracy'
        }
        
        for key, display_name in metric_display.items():
            mean_key = f"mean_{key}"
            if mean_key in results['summary_metrics']:
                mean_val = results['summary_metrics'][mean_key]
                min_val = results['summary_metrics'][f"min_{key}"]
                max_val = results['summary_metrics'][f"max_{key}"]
                print(f"{display_name:<20} {mean_val:.3f}    {min_val:.3f}    {max_val:.3f}")
        
        if 'mean_is_two_word_swap' in results['summary_metrics']:
            swap_rate = results['summary_metrics']['mean_is_two_word_swap']
            print(f"{'Two-word Swaps':<20} {swap_rate:.3f}    -        -")
    
    print("="*70)
    
    # Save results
    if not args.no_save:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate filename with timestamp if not specified
        if args.output:
            output_file = args.output
        else:
            output_file = f"results_{args.solver}_{args.model}_{timestamp}.json"
        
        # Add metadata to results
        results["metadata"] = {
            "solver": args.solver,
            "model": args.model,
            "test_file": args.test_file,
            "timestamp": datetime.now().isoformat(),
            "mode": "api" if args.use_api else "dummy",
            "limit": args.limit
        }
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {output_file}")
        
        # Update experiment history CSV
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
                'total_acc': results['accuracy'],
                'group_acc': group_acc,
                'correct': results['correct'],
                'total': results['total_examples'],
                'test_file': os.path.basename(args.test_file),
                'mode': 'api' if args.use_api else 'dummy'
            })
        
        print(f"📊 Updated experiment history: {history_file}")
    
    # Return exit code based on success (useful for CI/CD)
    return 0 if results['accuracy'] > 0 else 1


if __name__ == "__main__":
    main()