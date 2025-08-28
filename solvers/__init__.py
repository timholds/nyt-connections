from .base import BaseSolver
from .baseline import BaselineSolver
from .few_shot import FewShotSolver
from .dspy_solver import DSPySolver

SOLVER_REGISTRY = {
    "baseline": BaselineSolver,
    "few-shot": FewShotSolver,
    "dspy": DSPySolver,
}

def get_solver(solver_name: str) -> BaseSolver:
    """Get a solver instance by name."""
    if solver_name not in SOLVER_REGISTRY:
        available = ", ".join(SOLVER_REGISTRY.keys())
        raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {available}")
    
    solver_class = SOLVER_REGISTRY[solver_name]
    return solver_class()