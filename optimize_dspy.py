"""
Optimize DSPy solver for NYT Connections using MIPRO.
This script trains the prompt templates and example selection.
"""

import json
import os
from typing import List, Dict, Any, Set, Tuple
import dspy
from dspy.teleprompt import MIPROv2
from solvers.dspy_solver import ConnectionsSolver, ConnectionsSignature
from solvers.models import PuzzleSolution, GroupSolution
import random


def normalize_groups(groups: List[Set[str]]) -> List[Set[str]]:
    """Normalize groups by converting to lowercase sets."""
    return [set(w.lower().strip() for w in group) for group in groups]


def count_matching_groups(pred_groups: List[Set[str]], true_groups: List[Set[str]]) -> int:
    """
    Count how many predicted groups match the true groups exactly.
    Order doesn't matter for either groups or words within groups.
    """
    pred_normalized = normalize_groups(pred_groups)
    true_normalized = normalize_groups(true_groups)
    
    matches = 0
    matched_true = set()
    
    for pred_group in pred_normalized:
        for i, true_group in enumerate(true_normalized):
            if i not in matched_true and pred_group == true_group:
                matches += 1
                matched_true.add(i)
                break
    
    return matches


def puzzle_accuracy_metric(example: dspy.Example, pred, trace=None) -> float:
    """
    Metric for evaluating puzzle solutions.
    Returns 1.0 for perfect match, partial credit for correct groups.
    """
    try:
        # Parse true groups from example
        true_groups = []
        for i in range(1, 5):
            group_key = f"group{i}_words"
            if hasattr(example, group_key):
                words = getattr(example, group_key)
                if isinstance(words, str):
                    words_list = [w.strip() for w in words.split(",")]
                else:
                    words_list = list(words)
                true_groups.append(set(words_list))
        
        # Parse predicted groups
        pred_groups = []
        for i in range(1, 5):
            group_key = f"group{i}_words"
            if hasattr(pred, group_key):
                words = getattr(pred, group_key)
                if isinstance(words, str):
                    words_list = [w.strip() for w in words.split(",")]
                else:
                    words_list = list(words)
                pred_groups.append(set(words_list))
        
        # Count matching groups
        if len(pred_groups) != 4 or len(true_groups) != 4:
            return 0.0
        
        correct_groups = count_matching_groups(pred_groups, true_groups)
        
        # Return score (0.25 per correct group)
        score = correct_groups / 4.0
        
        # Bonus for getting all 4 correct (perfect solution)
        if correct_groups == 4:
            score = 1.0
        
        return score
        
    except Exception as e:
        print(f"Error in metric calculation: {e}")
        return 0.0


def load_examples(filepath: str = "examples.jsonl") -> List[dspy.Example]:
    """Load and format examples from JSONL file."""
    examples = []
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Format as DSPy example
            words_str = ", ".join(data['words'])
            groups = data['solution']['groups']
            
            # Build reasoning based on the solution
            reasoning_parts = [
                "Let me analyze these words systematically:",
                "",
                "Looking for different types of connections:",
                f"- Literal categories: {groups[0]['reason']}",
                f"- Wordplay/patterns: checking for homophones, palindromes, rhymes",
                f"- Phrases/idioms: words that commonly go together",
                f"- Pop culture: brands, geography, entertainment",
                "",
                "After careful analysis, I've identified these four groups:"
            ]
            reasoning = "\n".join(reasoning_parts)
            
            example = dspy.Example(
                words=words_str,
                reasoning=reasoning,
                group1_words=", ".join(groups[0]['words']),
                group1_reason=groups[0]['reason'],
                group2_words=", ".join(groups[1]['words']),
                group2_reason=groups[1]['reason'],
                group3_words=", ".join(groups[2]['words']),
                group3_reason=groups[2]['reason'],
                group4_words=", ".join(groups[3]['words']),
                group4_reason=groups[3]['reason']
            ).with_inputs('words')
            
            examples.append(example)
    
    return examples


def split_data(examples: List[dspy.Example], train_ratio: float = 0.8, seed: int = 42):
    """Split examples into train and validation sets."""
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_idx]
    val_set = shuffled[split_idx:]
    
    return train_set, val_set


def evaluate_solver(solver, test_set: List[dspy.Example], verbose: bool = False) -> Tuple[float, Dict]:
    """Evaluate solver on test set."""
    total_score = 0
    perfect_puzzles = 0
    group_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for i, example in enumerate(test_set):
        try:
            pred = solver(words=example.words)
            score = puzzle_accuracy_metric(example, pred)
            total_score += score
            
            # Count perfect solutions
            if score == 1.0:
                perfect_puzzles += 1
            
            # Track group distribution
            correct_groups = int(score * 4)
            group_distribution[correct_groups] += 1
            
            if verbose and i < 3:  # Show first 3 examples
                print(f"\nExample {i+1}:")
                print(f"Words: {example.words[:50]}...")
                print(f"Score: {score:.2f} ({correct_groups}/4 groups correct)")
                
        except Exception as e:
            print(f"Error evaluating example {i}: {e}")
            group_distribution[0] += 1
    
    avg_score = total_score / len(test_set) if test_set else 0
    
    stats = {
        'average_score': avg_score,
        'perfect_puzzles': perfect_puzzles,
        'perfect_rate': perfect_puzzles / len(test_set) if test_set else 0,
        'group_distribution': group_distribution
    }
    
    return avg_score, stats


def main():
    """Main optimization script."""
    print("=" * 80)
    print("DSPy NYT Connections Solver Optimization")
    print("=" * 80)
    
    # Load examples
    print("\n1. Loading examples...")
    examples = load_examples("examples.jsonl")
    print(f"   Loaded {len(examples)} examples")
    
    # Split data
    print("\n2. Splitting data...")
    train_set, val_set = split_data(examples, train_ratio=0.8)
    print(f"   Train: {len(train_set)} examples")
    print(f"   Validation: {len(val_set)} examples")
    
    # Initialize DSPy with OpenAI
    print("\n3. Initializing DSPy with OpenAI...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    lm = dspy.LM(
        model="gpt-4o-mini",
        api_key=api_key,
        max_tokens=800,
        temperature=0.7
    )
    dspy.settings.configure(lm=lm)
    
    # Create base solver
    print("\n4. Creating base solver...")
    solver = ConnectionsSolver()
    
    # Evaluate base solver
    print("\n5. Evaluating base solver on validation set...")
    base_score, base_stats = evaluate_solver(solver, val_set[:10], verbose=True)
    print(f"\n   Base solver average score: {base_score:.3f}")
    print(f"   Perfect puzzles: {base_stats['perfect_puzzles']}/{10} ({base_stats['perfect_rate']*100:.1f}%)")
    print(f"   Group distribution: {base_stats['group_distribution']}")
    
    # Set up optimizer
    print("\n6. Setting up MIPRO optimizer...")
    print("   This will optimize:")
    print("   - Signature field descriptions")
    print("   - Chain-of-thought prompt template")
    print("   - Example selection strategy")
    
    optimizer = MIPROv2(
        metric=puzzle_accuracy_metric,
        num_candidates=10,  # Number of prompt variations to try
        init_temperature=0.7,
        verbose=True
    )
    
    # Optimize the solver
    print("\n7. Running optimization (this may take a while)...")
    print("   Using subset of training data to reduce API calls...")
    
    # Use subset for optimization to reduce costs
    train_subset = train_set[:50]  # Use 50 examples for optimization
    val_subset = val_set[:20]  # Use 20 for validation during optimization
    
    try:
        optimized_solver = optimizer.compile(
            solver,
            trainset=train_subset,
            valset=val_subset,
            num_trials=15,  # Number of optimization trials
            max_bootstrapped_demos=3,  # Max examples to include
            max_labeled_demos=3
        )
        
        print("\n8. Optimization complete!")
        
        # Evaluate optimized solver
        print("\n9. Evaluating optimized solver on full validation set...")
        opt_score, opt_stats = evaluate_solver(optimized_solver, val_set[:20], verbose=True)
        print(f"\n   Optimized solver average score: {opt_score:.3f}")
        print(f"   Perfect puzzles: {opt_stats['perfect_puzzles']}/{20} ({opt_stats['perfect_rate']*100:.1f}%)")
        print(f"   Group distribution: {opt_stats['group_distribution']}")
        
        # Compare improvement
        print("\n10. Results Summary:")
        print("   " + "-" * 50)
        print(f"   Base solver score:      {base_score:.3f}")
        print(f"   Optimized solver score: {opt_score:.3f}")
        print(f"   Improvement:            {(opt_score - base_score):.3f} ({((opt_score/base_score - 1)*100):.1f}%)")
        
        # Save optimized solver
        print("\n11. Saving optimized solver...")
        optimized_solver.save("optimized_connections_solver.json")
        print("    Saved to optimized_connections_solver.json")
        
        # Show optimized prompt if available
        if hasattr(optimized_solver.generate, 'extended_signature'):
            print("\n12. Optimized Signature:")
            print("   " + "-" * 50)
            sig = optimized_solver.generate.extended_signature
            for field_name, field in sig.fields.items():
                if hasattr(field, 'json_schema_extra') and field.json_schema_extra:
                    desc = field.json_schema_extra.get('desc', 'No description')
                    print(f"   {field_name}: {desc[:100]}...")
        
    except Exception as e:
        print(f"\n   Optimization failed: {e}")
        print("   This might be due to API rate limits or costs.")
        print("   Consider reducing num_candidates, num_trials, or dataset size.")
        
        # Create a manually optimized version as fallback
        print("\n   Creating manually optimized solver instead...")
        optimized_solver = create_manual_optimized_solver()
        
        # Evaluate manual optimization
        opt_score, opt_stats = evaluate_solver(optimized_solver, val_set[:20], verbose=False)
        print(f"   Manually optimized score: {opt_score:.3f}")
    
    print("\n" + "=" * 80)
    print("Optimization complete!")
    print("=" * 80)


def create_manual_optimized_solver():
    """Create a manually optimized solver with better prompts."""
    
    class OptimizedConnectionsSignature(dspy.Signature):
        """Solve NYT Connections by finding 4 groups of 4 related words each."""
        
        words = dspy.InputField(
            desc="16 words to organize into 4 distinct categories. Each word belongs to exactly one group."
        )
        reasoning = dspy.OutputField(
            desc="Step-by-step analysis: 1) List all 16 words. 2) Identify potential categories (literal, wordplay, phrases, pop culture). 3) Test groupings ensuring no word appears twice. 4) Verify each group has exactly 4 words."
        )
        group1_words = dspy.OutputField(
            desc="Exactly 4 words forming the first category, separated by commas. Check: palindromes, rhymes, homophones, or literal categories."
        )
        group1_reason = dspy.OutputField(
            desc="The specific connection between these 4 words (be precise - e.g., 'Types of pasta' not just 'food')"
        )
        group2_words = dspy.OutputField(
            desc="Exactly 4 different words forming the second category, separated by commas. Cannot reuse any words from group1."
        )
        group2_reason = dspy.OutputField(
            desc="The specific connection for group 2 (check for: weather phenomena, brand names, idioms)"
        )
        group3_words = dspy.OutputField(
            desc="Exactly 4 different words forming the third category, separated by commas. Cannot reuse words from groups 1-2."
        )
        group3_reason = dspy.OutputField(
            desc="The specific connection for group 3 (consider: geography, sports, entertainment references)"
        )
        group4_words = dspy.OutputField(
            desc="The remaining 4 words forming the fourth category, separated by commas. Must be the 4 words not used in groups 1-3."
        )
        group4_reason = dspy.OutputField(
            desc="The specific connection for group 4 (often the trickiest - look for subtle wordplay or uncommon categories)"
        )
    
    class OptimizedConnectionsSolver(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(OptimizedConnectionsSignature)
        
        def forward(self, words: str):
            return self.generate(words=words)
    
    return OptimizedConnectionsSolver()


if __name__ == "__main__":
    main()