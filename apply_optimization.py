"""
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
