"""
Apply MIPRO optimizations to DSPy solvers.
This script demonstrates how to use the optimized components from optimize_once.py

The optimization process creates:
1. optimized_solver_{model}.json - Complete optimized solver with instructions and demos
2. optimized_prompts_{model}.json - Field descriptions (if extracted)
3. optimized_demos_{model}.json - Best few-shot examples (if created)

This script shows how to:
- Load and use the complete optimized solver
- Apply optimized prompts to existing signatures
- Use optimized demos in your own solvers
"""

import json
import os
import dspy
from typing import Optional, Dict, Any


def get_optimized_solver(model: str = "gpt-5-mini") -> Optional[dspy.Module]:
    """
    Load a complete optimized solver for a specific model.
    
    Args:
        model: The model name to load optimizations for
    
    Returns:
        Optimized DSPy module or None if not found
    """
    model_suffix = model.replace("-", "_")
    solver_file = f"optimized_solver_{model_suffix}.json"
    
    if not os.path.exists(solver_file):
        print(f"No optimized solver found at {solver_file}")
        print(f"Run: python optimize_once.py --model {model}")
        return None
    
    print(f"Loading optimized solver from {solver_file}")
    solver = dspy.Module()
    solver.load(solver_file)
    return solver


def get_optimized_prompts(model: str = "gpt-5-mini") -> Dict[str, str]:
    """
    Load optimized field descriptions from MIPRO optimization.
    
    Args:
        model: The model name to load prompts for
    
    Returns:
        Dictionary of field names to optimized descriptions
    """
    model_suffix = model.replace("-", "_")
    prompts_file = f"optimized_prompts_{model_suffix}.json"
    
    if not os.path.exists(prompts_file):
        print(f"No optimized prompts found at {prompts_file}")
        return {}
    
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded optimized prompts from {prompts_file}")
    print(f"  Baseline score: {data.get('baseline_score', 'N/A'):.3f}")
    print(f"  Optimized score: {data.get('optimized_score', 'N/A'):.3f}")
    print(f"  Improvement: +{data.get('improvement', 0):.3f}")
    
    return data.get('field_descriptions', {})


def get_optimized_demos(model: str = "gpt-5-mini") -> list:
    """
    Load the best few-shot examples selected by MIPRO.
    
    Args:
        model: The model name to load demos for
    
    Returns:
        List of demonstration examples
    """
    model_suffix = model.replace("-", "_")
    demos_file = f"optimized_demos_{model_suffix}.json"
    
    if not os.path.exists(demos_file):
        print(f"No optimized demos found at {demos_file}")
        return []
    
    with open(demos_file, 'r') as f:
        demos = json.load(f)
    
    print(f"Loaded {len(demos)} optimized demos from {demos_file}")
    return demos


def create_optimized_signature(model: str = "gpt-5-mini") -> type:
    """
    Create an optimized ConnectionsSignature using MIPRO-optimized prompts.
    
    Args:
        model: The model name to use prompts for
    
    Returns:
        Optimized signature class
    """
    prompts = get_optimized_prompts(model)
    
    if not prompts:
        # Fallback to default signature
        from solvers.dspy_solver import ConnectionsSignature
        return ConnectionsSignature
    
    class OptimizedConnectionsSignature(dspy.Signature):
        """Solve NYT Connections using MIPRO-optimized prompts."""
        
        words = dspy.InputField(
            desc=prompts.get('words', "16 words to group into 4 categories")
        )
        reasoning = dspy.OutputField(
            desc=prompts.get('reasoning', "Step-by-step reasoning exploring different grouping strategies")
        )
        group1_words = dspy.OutputField(
            desc=prompts.get('group1_words', "First group of exactly 4 words (comma-separated)")
        )
        group1_reason = dspy.OutputField(
            desc=prompts.get('group1_reason', "Connecting theme for group 1")
        )
        group2_words = dspy.OutputField(
            desc=prompts.get('group2_words', "Second group of exactly 4 words (comma-separated)")
        )
        group2_reason = dspy.OutputField(
            desc=prompts.get('group2_reason', "Connecting theme for group 2")
        )
        group3_words = dspy.OutputField(
            desc=prompts.get('group3_words', "Third group of exactly 4 words (comma-separated)")
        )
        group3_reason = dspy.OutputField(
            desc=prompts.get('group3_reason', "Connecting theme for group 3")
        )
        group4_words = dspy.OutputField(
            desc=prompts.get('group4_words', "Fourth group of exactly 4 words (comma-separated)")
        )
        group4_reason = dspy.OutputField(
            desc=prompts.get('group4_reason', "Connecting theme for group 4")
        )
    
    return OptimizedConnectionsSignature


def apply_optimizations_to_solver(solver: dspy.Module, model: str = "gpt-5-mini") -> dspy.Module:
    """
    Apply MIPRO optimizations to an existing solver.
    
    Args:
        solver: The solver module to optimize
        model: The model name to use optimizations for
    
    Returns:
        The solver with optimizations applied
    """
    # Get optimized components
    OptimizedSig = create_optimized_signature(model)
    demos = get_optimized_demos(model)
    
    # Apply optimized signature
    if hasattr(solver, 'generate'):
        solver.generate = dspy.ChainOfThought(OptimizedSig)
        
        # Apply optimized demos if available
        if demos:
            # Convert demos to DSPy Example format
            dspy_demos = []
            for demo in demos:
                example = dspy.Example(
                    words=demo.get('words', ''),
                    reasoning=demo.get('reasoning', ''),
                    group1_words=demo.get('group1_words', ''),
                    group1_reason=demo.get('group1_reason', ''),
                    group2_words=demo.get('group2_words', ''),
                    group2_reason=demo.get('group2_reason', ''),
                    group3_words=demo.get('group3_words', ''),
                    group3_reason=demo.get('group3_reason', ''),
                    group4_words=demo.get('group4_words', ''),
                    group4_reason=demo.get('group4_reason', '')
                ).with_inputs('words')
                dspy_demos.append(example)
            
            solver.generate.demos = dspy_demos
            print(f"Applied {len(dspy_demos)} optimized demos to solver")
    
    return solver


# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply MIPRO optimizations to DSPy solvers")
    parser.add_argument("--model", type=str, default="gpt-5-mini",
                        help="Model to load optimizations for")
    parser.add_argument("--test", action="store_true",
                        help="Test the optimized solver with a sample puzzle")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Loading optimizations for {args.model}")
    print(f"{'='*60}\n")
    
    # Option 1: Load complete optimized solver
    optimized_solver = get_optimized_solver(args.model)
    if optimized_solver:
        print("\n✓ Complete optimized solver loaded successfully")
    
    # Option 2: Load individual components
    prompts = get_optimized_prompts(args.model)
    demos = get_optimized_demos(args.model)
    
    if prompts:
        print(f"\n✓ Loaded {len(prompts)} optimized field descriptions")
    
    if demos:
        print(f"✓ Loaded {len(demos)} optimized demonstration examples")
    
    # Option 3: Create optimized signature class
    OptimizedSig = create_optimized_signature(args.model)
    print(f"\n✓ Created optimized signature class: {OptimizedSig.__name__}")
    
    # Test with a sample puzzle if requested
    if args.test and optimized_solver:
        print(f"\n{'='*60}")
        print("Testing optimized solver with sample puzzle")
        print(f"{'='*60}\n")
        
        # Sample puzzle words
        test_words = "BANK, RIVER, STREAM, CURRENT, FLOW, TIDE, WAVE, DRIFT, CASH, MONEY, FUNDS, CAPITAL, LEAN, TILT, SLOPE, ANGLE"
        
        # Initialize DSPy with the model
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            lm = dspy.LM(
                model=args.model,
                api_key=api_key,
                max_tokens=800,
                temperature=0.7
            )
            dspy.settings.configure(lm=lm)
            
            # Run the solver
            result = optimized_solver(words=test_words)
            
            print("Results:")
            for i in range(1, 5):
                words = getattr(result, f"group{i}_words", "")
                reason = getattr(result, f"group{i}_reason", "")
                print(f"\nGroup {i}: {words}")
                print(f"  Reason: {reason}")
        else:
            print("OPENAI_API_KEY not set - skipping test")
    
    print(f"\n{'='*60}")
    print("To use these optimizations in your solver:")
    print(f"{'='*60}")
    print("""
from apply_optimization import get_optimized_solver, apply_optimizations_to_solver
from solvers.dspy_solver import ConnectionsSolver

# Option 1: Use complete optimized solver
solver = get_optimized_solver('gpt-5-mini')

# Option 2: Apply optimizations to existing solver
my_solver = ConnectionsSolver()
my_solver = apply_optimizations_to_solver(my_solver, 'gpt-5-mini')
""")