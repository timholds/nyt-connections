# NYT Connections Solver

A solver for NYT Connections puzzles using word embeddings and semantic similarity.

## Embedding Storage & Performance

The system generates embeddings for both individual **words** (from puzzles) and **themes** (category descriptions like "nba teams" or "palindromes"). These are stored separately since they serve different semantic purposes - themes represent abstract concepts while words are concrete tokens. 

Key design choices:

1. **Unnormalized Storage**: Embeddings are stored with their original magnitudes from OpenAI, not normalized to unit vectors. This preserves information useful for outlier detection and clustering algorithms.

2. **Dual Format Storage**: Maintains embeddings as both a dictionary (for compatibility) and a NumPy matrix (for speed). This trades ~30MB extra memory for 100-1000x faster operations.

3. **Vectorized Operations**: Uses NumPy matrix multiplication to compute all 256 pairwise similarities for a 16-word puzzle in a single operation (~10 microseconds vs ~10 milliseconds).

4. **Float64 Precision**: Uses double precision for better numerical stability in vector math operations.

The pickle format loads in ~0.01 seconds compared to ~1-2 seconds for JSON, making iterative development much faster. For our ~4,000 unique words and ~1,600 themes with 1,536-dimensional embeddings, the memory footprint is about 48MB, which easily fits in RAM while enabling instant lookups.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Get embeddings (one-time, ~$0.10)
# This creates both JSON and optimized pickle formats automatically
python get_embeddings.py
```

## LLM API Design Choice

The solver uses OpenAI's structured output (`beta.chat.completions.parse`) to ensure reliable, validated responses with reasoning for each group. While this requires models that support structured output (gpt-4o-mini or newer), it guarantees valid JSON responses matching our Pydantic schema and provides meaningful explanations for each grouping. This is crucial for understanding the model's reasoning and debugging incorrect groupings. The cost remains very low with gpt-4o-mini at $0.15/1M input tokens and $0.60/1M output tokens - running on 400 puzzles costs less than $0.20 total.

## Pydantic Validation with Retry Logic

The solver enforces strict puzzle constraints using Pydantic models with custom validators:
- **Exactly 4 groups** of 4 words each
- **All 16 words used once** - no duplicates, no omissions
- **Reasoning required** for each grouping

The system prompt in baseline.py to explicitly states the critical rules. However, they aren't always met. When validation fails, we have a mechanism in place to retry up to 3 times with specific feedback about what went wrong. For example, if the LLM uses a word twice, it gets told: "You included the same word in multiple groups. Each word must appear in EXACTLY ONE group."
- Note: only retries on validation errors which we have written Pydantic rules for, not other types of errors

The key insight is that Pydantic/OpenAI structured outputs guarantee valid JSON structure but not our custom business rules. When validation fails, we now catch the specific error and give the LLM targeted feedback about what went wrong, increasing the chance it will fix the issue on retry.


## Validation Metrics

The scoring system (`score.py`) implements comprehensive metrics to understand model performance beyond simple accuracy:

- **Exact Match & Group Accuracy**: Track perfect solutions and number of correct groups (0-4), identifying if models consistently get 3/4 groups right with one problematic category
- **Word Placement & Pairwise Accuracy**: Measure fine-grained performance - whether individual words and word pairs are correctly grouped, helping distinguish partial understanding from random guessing
- **Two-word Swap Detection**: Identifies near-misses where just 2 words are swapped between otherwise correct groups, revealing if the model understands the categories but makes minor boundary errors
- **Average Group Overlap**: Shows how close incorrect groups are to being right, useful for understanding whether wrong answers still capture some semantic similarity

These metrics help distinguish systematic errors from random failures and guide improvements - for example, high pairwise accuracy with low exact match suggests the model understands relationships but struggles with precise boundaries between similar categories.

## Files

- `examples.jsonl` - 399 puzzle examples with solutions
- `get_embeddings.py` - Fetches embeddings from OpenAI (includes both words and themes)
- `embedding_store.py` - Optimized storage class for fast lookups
- `embedding_utils.py` - Utility functions for similarity and clustering
- `solve.py` - Main solver implementation
- `score.py` - Comprehensive scoring system with detailed metrics