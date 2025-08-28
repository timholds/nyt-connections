from typing import Optional, List, Dict, Any
from .base import BaseSolver
from .baseline import BaselineSolver
from .few_shot import FewShotSolver
from .dspy_solver import DSPySolver
from .multi_stage_solver import MultiStageSolver
from .hungarian_solver import HungarianSolver
from .constraint_solver import ConstraintValidationSolver

SOLVER_REGISTRY = {
    "baseline": BaselineSolver,
    "few-shot": FewShotSolver,
    "dspy": DSPySolver,
    "multi-stage": MultiStageSolver,
    "hungarian": HungarianSolver,
    "constraint": ConstraintValidationSolver,
}

def get_solver(solver_name: str, examples: Optional[List[Dict[str, Any]]] = None) -> BaseSolver:
    """Get a solver instance by name.
    
    Args:
        solver_name: Name of the solver to instantiate
        examples: Optional list of example puzzles for few-shot/dspy solvers
    """
    if solver_name not in SOLVER_REGISTRY:
        available = ", ".join(SOLVER_REGISTRY.keys())
        raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {available}")
    
    solver_class = SOLVER_REGISTRY[solver_name]
    
    # Pass appropriate arguments to each solver
    if solver_name in ["few-shot", "dspy", "multi-stage", "constraint"]:
        return solver_class(examples=examples)
    elif solver_name == "hungarian":
        # Use the optimized pickle file for faster loading
        return solver_class(embeddings_path="word_theme_text-embedding-3-large.pkl")
    else:
        return solver_class()