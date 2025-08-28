"""
Optimize DSPy prompts once with MIPRO and apply to all solvers.
This script runs expensive optimization once and saves reusable components.
"""

import json
import os
from typing import List, Dict, Any, Set, Tuple
import dspy
from dspy.teleprompt import MIPROv2
from solvers.dspy_solver import ConnectionsSolver, ConnectionsSignature
import random
import pickle
import litellm


def normalize_groups(groups: List[Set[str]]) -> List[Set[str]]:
    """Normalize groups by converting to lowercase sets."""
    return [set(w.lower().strip() for w in group) for group in groups]


def puzzle_accuracy_metric(example: dspy.Example, pred, trace=None) -> float:
    """
    Metric for evaluating puzzle solutions.
    Order of groups and words within groups doesn't matter.
    """
    try:
        # Parse true groups
        true_groups = []
        for i in range(1, 5):
            group_key = f"group{i}_words"
            if hasattr(example, group_key):
                words = getattr(example, group_key)
                if isinstance(words, str):
                    words_list = [w.strip() for w in words.split(",")]
                else:
                    words_list = list(words)
                true_groups.append(set(w.lower() for w in words_list))
        
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
                pred_groups.append(set(w.lower() for w in words_list))
        
        # Count matching groups (order-independent)
        matches = 0
        matched_true = set()
        
        for pred_group in pred_groups:
            for j, true_group in enumerate(true_groups):
                if j not in matched_true and pred_group == true_group:
                    matches += 1
                    matched_true.add(j)
                    break
        
        # Score: 0.25 per correct group, bonus for perfect
        score = matches / 4.0
        if matches == 4:
            score = 1.0  # Perfect bonus
            
        return score
        
    except Exception as e:
        print(f"Error in metric: {e}")
        return 0.0


def load_examples(filepath: str = "examples.jsonl") -> List[dspy.Example]:
    """Load and format examples from JSONL file."""
    examples = []
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            words_str = ", ".join(data['words'])
            groups = data['solution']['groups']
            
            # Simple reasoning template
            reasoning = f"Analyzing 16 words for 4 groups of 4. Looking for categories, wordplay, phrases, and cultural references."
            
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


def run_optimization(model: str = "gpt-5-mini"):
    """Run MIPRO optimization and save results.
    
    Args:
        model: The OpenAI model to optimize for (default: gpt-5-mini)
    """
    
    print("="*80)
    print("MIPRO Optimization for DSPy Connections Solver")
    print(f"Optimizing for model: {model}")
    print("="*80)
    
    # Load examples
    print("\n1. Loading examples...")
    examples = load_examples("examples.jsonl")
    print(f"   Loaded {len(examples)} examples")
    
    # Split data
    print("\n2. Splitting data...")
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.8)
    train_set = examples[:split_idx]
    val_set = examples[split_idx:]
    print(f"   Train: {len(train_set)} examples")
    print(f"   Validation: {len(val_set)} examples")
    
    # Initialize DSPy
    print("\n3. Initializing DSPy with OpenAI...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Check if this is a GPT-5 reasoning model
    if 'gpt-5' in model.lower():
        # GPT-5 models require specific parameters
        # Enable dropping unsupported params for GPT-5 compatibility
        litellm.drop_params = True
        
        lm = dspy.LM(
            model=model,
            api_key=api_key,
            max_tokens=20000,  # Required minimum for GPT-5
            temperature=1.0     # Required temperature for GPT-5
        )
    else:
        # Regular models use standard parameters
        lm = dspy.LM(
            model=model,
            api_key=api_key,
            max_tokens=800,
            temperature=0.7
        )
    dspy.settings.configure(lm=lm)
    
    # Create base solver
    print("\n4. Creating base solver...")
    solver = ConnectionsSolver()
    
    # Quick baseline evaluation
    print("\n5. Quick baseline test (5 examples)...")
    baseline_scores = []
    for example in val_set[:5]:
        try:
            pred = solver(words=example.words)
            score = puzzle_accuracy_metric(example, pred)
            baseline_scores.append(score)
            print(f"   Example score: {score:.2f}")
        except:
            baseline_scores.append(0.0)
    
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    print(f"   Baseline average: {baseline_avg:.3f}")
    
    # MIPRO Optimization
    print("\n6. Running MIPRO optimization...")
    print("   This will optimize:")
    print("   - Field descriptions in the signature")
    print("   - Chain-of-thought reasoning template")
    print("   - Few-shot example selection")
    print("\n   Using limited dataset to reduce costs:")
    print(f"   - Training on 30 examples")
    print(f"   - Validating on 10 examples")
    print(f"   - Running 10 optimization trials")
    
    # GPT-5 requires temperature=1.0, so we need to handle this specially
    if 'gpt-5' in model.lower():
        optimizer = MIPROv2(
            metric=puzzle_accuracy_metric,
            init_temperature=1.0,  # GPT-5 only supports temperature=1.0
            verbose=True
        )
    else:
        optimizer = MIPROv2(
            metric=puzzle_accuracy_metric,
            init_temperature=0.7,
            verbose=True
        )
    
    # Use smaller subset for cost efficiency
    train_subset = train_set[:30]
    val_subset = val_set[:10]
    
    print("\n   Starting optimization (this will take a few minutes)...")
    
    optimized_solver = optimizer.compile(
        solver,
        trainset=train_subset,
        valset=val_subset,
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    print("\n7. Optimization complete!")
    
    # Test optimized solver
    print("\n8. Testing optimized solver...")
    opt_scores = []
    for i, example in enumerate(val_set[:10]):
        try:
            pred = optimized_solver(words=example.words)
            score = puzzle_accuracy_metric(example, pred)
            opt_scores.append(score)
            print(f"   Example {i+1}: {score:.2f}")
        except Exception as e:
            print(f"   Example {i+1}: Failed - {e}")
            opt_scores.append(0.0)
    
    opt_avg = sum(opt_scores) / len(opt_scores) if opt_scores else 0
    
    print(f"\n   Results:")
    print(f"   Baseline: {baseline_avg:.3f}")
    print(f"   Optimized: {opt_avg:.3f}")
    print(f"   Improvement: +{(opt_avg - baseline_avg):.3f}")
    
    # Save optimization results
    print("\n9. Saving optimization artifacts...")
    
    # Create model-specific filename
    model_suffix = model.replace("-", "_")
    solver_filename = f"optimized_solver_{model_suffix}.json"
    
    # Save the optimized solver
    optimized_solver.save(solver_filename)
    print(f"   ✓ Saved optimized solver to {solver_filename}")
    
    # The optimized solver is already saved as JSON above
    # Now let's extract additional useful information from the saved file
    
    # Load back the saved JSON to extract instructions and demos
    with open(solver_filename, 'r') as f:
        saved_data = json.load(f)
    
    prompts_filename = None
    demos_filename = None
    
    # Extract optimized instructions/signature if available
    if 'generate.predict' in saved_data and 'signature' in saved_data['generate.predict']:
        signature_data = saved_data['generate.predict']['signature']
        
        prompts_filename = f"optimized_prompts_{model_suffix}.json"
        with open(prompts_filename, 'w') as f:
            json.dump({
                'model': model,
                'instructions': signature_data.get('instructions', ''),
                'fields': signature_data.get('fields', []),
                'baseline_score': baseline_avg,
                'optimized_score': opt_avg,
                'improvement': opt_avg - baseline_avg
            }, f, indent=2)
        
        print(f"   ✓ Saved optimized prompts to {prompts_filename}")
    
    # Extract demos if available
    if 'generate.predict' in saved_data and 'demos' in saved_data['generate.predict']:
        demos_data = saved_data['generate.predict']['demos']
        
        if demos_data:  # Only save if there are actual demos
            demos_filename = f"optimized_demos_{model_suffix}.json"
            with open(demos_filename, 'w') as f:
                json.dump(demos_data, f, indent=2)
            
            print(f"   ✓ Saved {len(demos_data)} optimized demos to {demos_filename}")
    
    print("\n" + "="*80)
    print("Optimization Complete!")
    print("="*80)
    print(f"\nOptimized components saved for model {model}:")
    print(f"1. {solver_filename} - Complete optimized solver")
    if prompts_filename:
        print(f"2. {prompts_filename} - Optimized field descriptions")
    if demos_filename:
        print(f"3. {demos_filename} - Best few-shot examples")
    print("\nSee apply_optimization.py for how to use these in your solvers.")
    
    return optimized_solver


def create_application_script():
    """Create a script showing how to apply optimizations."""
    
    script = '''"""
Apply MIPRO optimizations to all DSPy solvers.
This script shows how to use the optimized components.
"""

import json
import dspy
from solvers.dspy_solver import ConnectionsSignature
from solvers.multi_stage_solver import (
    ThemeAnalysisSignature,
    HypothesisGenerationSignature,
    ValidationSignature,
    RefinementSignature
)


def load_optimized_prompts():
    """Load the optimized prompts from MIPRO."""
    with open('optimized_prompts.json', 'r') as f:
        data = json.load(f)
    return data['field_descriptions']


def apply_to_connections_signature():
    """Apply optimized prompts to ConnectionsSignature."""
    prompts = load_optimized_prompts()
    
    class OptimizedConnectionsSignature(dspy.Signature):
        """Solve NYT Connections using MIPRO-optimized prompts."""
        
        # Apply optimized descriptions
        words = dspy.InputField(desc=prompts.get('words', ConnectionsSignature.words.desc))
        reasoning = dspy.OutputField(desc=prompts.get('reasoning', ConnectionsSignature.reasoning.desc))
        group1_words = dspy.OutputField(desc=prompts.get('group1_words', ConnectionsSignature.group1_words.desc))
        group1_reason = dspy.OutputField(desc=prompts.get('group1_reason', ConnectionsSignature.group1_reason.desc))
        group2_words = dspy.OutputField(desc=prompts.get('group2_words', ConnectionsSignature.group2_words.desc))
        group2_reason = dspy.OutputField(desc=prompts.get('group2_reason', ConnectionsSignature.group2_reason.desc))
        group3_words = dspy.OutputField(desc=prompts.get('group3_words', ConnectionsSignature.group3_words.desc))
        group3_reason = dspy.OutputField(desc=prompts.get('group3_reason', ConnectionsSignature.group3_reason.desc))
        group4_words = dspy.OutputField(desc=prompts.get('group4_words', ConnectionsSignature.group4_words.desc))
        group4_reason = dspy.OutputField(desc=prompts.get('group4_reason', ConnectionsSignature.group4_reason.desc))
    
    return OptimizedConnectionsSignature


def create_optimized_solver():
    """Create a solver using the fully optimized module."""
    solver = dspy.Module()
    solver.load("optimized_solver.json")
    return solver


def apply_learnings_to_multistage():
    """Apply MIPRO learnings to MultiStage solver signatures."""
    prompts = load_optimized_prompts()
    
    # Extract key insights from optimized reasoning prompt
    reasoning_insights = prompts.get('reasoning', '')
    
    # Apply insights to theme analysis
    class OptimizedThemeAnalysisSignature(dspy.Signature):
        """Analyze using MIPRO-learned strategies."""
        
        words = dspy.InputField(desc=prompts.get('words', '16 words to analyze'))
        literal_themes = dspy.OutputField(
            desc="Categories like those found in optimized groups: " + prompts.get('group1_reason', '')[:100]
        )
        wordplay_patterns = dspy.OutputField(
            desc="Wordplay patterns emphasized in optimization: palindromes, homophones, rhymes"
        )
        phrase_patterns = dspy.OutputField(
            desc="Common phrases as discovered in optimization"
        )
        cultural_references = dspy.OutputField(
            desc="Pop culture patterns from optimized examples"
        )
        analysis_summary = dspy.OutputField(
            desc="Summary using MIPRO-optimized reasoning approach"
        )
    
    return OptimizedThemeAnalysisSignature


# Usage in your solvers:
if __name__ == "__main__":
    # Option 1: Load complete optimized solver
    optimized = create_optimized_solver()
    
    # Option 2: Apply optimized prompts to existing signatures
    OptimizedConnections = apply_to_connections_signature()
    
    # Option 3: Apply learnings to other solvers
    OptimizedThemeAnalysis = apply_learnings_to_multistage()
    
    print("Optimizations applied successfully!")
    print("Use these optimized signatures in your solver classes.")
'''
    
    with open('apply_optimization.py', 'w') as f:
        f.write(script)
    
    print("Created apply_optimization.py showing how to use optimizations")


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Optimize DSPy solver using MIPRO")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        choices=["gpt-5-mini", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                        help="OpenAI model to optimize for (default: gpt-5-mini)")
    args = parser.parse_args()
    
    # Run the optimization for the specified model
    optimized_solver = run_optimization(model=args.model)
    
    # Create application script
    create_application_script()