# NYT Connections Solver

A word puzzle solver that partitions 16 words into 4 groups of logically-related words using LLMs.

## Setup Instructions

1. **Create and activate virtual environment**
```bash
uv venv puzzle-env --python 3.12
source puzzle-env/bin/activate
```

2. **Install dependencies**
```bash
uv pip install -r requirements.txt
```
Note: If `uv` is not installed, use `pip install -r requirements.txt`

3. **Set OpenAI API key**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

4. **Run the solver**
```bash
python solve.py --file examples.jsonl --solver dspy --use-api --limit 500 --model gpt-5-mini
```

5. **Optional: Generate embeddings** (only needed for Hungarian solver, ~$0.10)
```bash
python get_embeddings.py
```

The solver automatically grades solutions against ground truth by default.

### Testing Different Solvers
Multiple solver approaches with different quality/cost/latency tradeoffs:

```bash
# Baseline solver (simplest, fewest tokens)
python solve.py --solver baseline --use-api  --limit 500

# Few-shot solver with static examples
python solve.py --solver few-shot --use-api  --limit 500

# DSPy solver with optimized examples (best accuracy)
python solve.py --solver dspy --use-api  --limit 500

# Multi-stage reasoning pipeline (more expensive)
python solve.py --solver multi-stage --use-api --limit 500

# Hungarian algorithm solver (deterministic, requires embeddings)
python solve.py --solver hungarian --use-api  --limit 500

# Constraint-based solver (hybrid approach)
python solve.py --solver constraint --use-api --limit 500

# Compare all solvers at once
python solve.py --all --use-api --limit 500
```

### Additional Parameters
```bash

# Use different models
python solve.py --model gpt-4o
python solve.py --model gpt-4o-mini  # default, most cost-effective
python solve.py --model gpt-5-mini 

# Skip automatic grading (just show solutions)
python solve.py --no-score

# Use dummy responses for testing (no API calls)
python solve.py  # without --use-api flag
```

## Viewing Experiment History

After running experiments, view and compare results:

```bash
# View experiment history and statistics
python score.py --compare
```

## Key Files

- `solve.py` - Main solver with automatic grading
- `score.py` - View experiment history and statistics
- `solvers/` - Modular solver implementations
  - `baseline.py` - Simple prompt-based solver
  - `few_shot.py` - Solver with static examples
  - `dspy_solver.py` - DSPy-based solver with dynamic example selection
  - `multi_stage_solver.py` - Multi-stage reasoning pipeline
  - `hungarian_solver.py` - Deterministic similarity-based algorithm
  - `constraint_solver.py` - Constraint programming approach
- `get_embeddings.py` - Fetches and caches OpenAI embeddings
- `requirements.txt` - Python dependencies

## Metrics

The solver automatically reports:
- **Exact Match**: Percentage of perfectly solved puzzles
- **Group Accuracy**: Average number of correct groups (0-4)
- **Pairwise Accuracy**: Percentage of word pairs correctly grouped together
- **Near Misses**: Puzzles with only 2 words swapped between groups