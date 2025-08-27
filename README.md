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

## Files

- `examples.jsonl` - 399 puzzle examples with solutions
- `get_embeddings.py` - Fetches embeddings from OpenAI (includes both words and themes)
- `embedding_store.py` - Optimized storage class for fast lookups
- `embedding_utils.py` - Utility functions for similarity and clustering
- `solve.py` - Main solver implementation