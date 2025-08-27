import json
import os
import argparse
from typing import List, Dict
from datetime import datetime
from openai import OpenAI
from embedding_utils import EmbeddingStore

def load_unique_words_and_themes(filename: str) -> tuple:
    """Load all unique words and themes from the examples JSONL file.
       Note: Themes are multi-word phrases (e.g., 'nba teams', 'units of length')
       that need to be embedded as complete phrases, not individual words."""
    unique_words = set()
    unique_themes = set()
    
    with open(filename, 'r') as f:
        for line in f:
            example = json.loads(line)
            unique_words.update(example['words'])
            # Extract complete theme phrases from solution groups
            for group in example['solution']['groups']:
                unique_themes.add(group['reason'])
    
    return unique_words, unique_themes

def log_api_cost(cost: float, model: str, num_items: int, item_type: str):
    """Log API cost to a tracking file."""
    log_file = "openai_api_costs.txt"
    
    # Read existing total if file exists
    total_cost = 0
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if line.startswith("TOTAL:"):
                        total_cost = float(line.split('$')[1])
                        break
        except:
            pass
    
    # Add new cost
    total_cost += cost
    
    # Append new entry and update total
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"{timestamp} | Model: {model} | {num_items} {item_type} | Cost: ${cost:.4f}\n")
        f.write(f"TOTAL: ${total_cost:.4f}\n")
        f.write("-" * 70 + "\n")
    
    return total_cost

def get_embeddings_batch(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> tuple:
    """Get embeddings for a batch of texts from OpenAI and return (embeddings, tokens_used)."""
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    # Extract token usage from response
    tokens_used = response.usage.total_tokens
    embeddings = [item.embedding for item in response.data]
    return embeddings, tokens_used

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate embeddings for NYT Connections words and themes')
    parser.add_argument('--model', type=str, default='text-embedding-3-small',
                        help='OpenAI embedding model to use (default: text-embedding-3-small)')
    args = parser.parse_args()
    embedding_model = args.model
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Load unique words and themes
    print("Loading words and themes from examples.jsonl...")
    unique_words, unique_themes = load_unique_words_and_themes("examples.jsonl")
    words_list = list(unique_words)  
    themes_list = list(unique_themes) 
    
    # Count total word occurrences for statistics
    total_occurrences = 0
    word_counts = {}
    with open("examples.jsonl", 'r') as f:
        for line in f:
            example = json.loads(line)
            total_occurrences += len(example['words'])
            for word in example['words']:
                word_counts[word] = word_counts.get(word, 0) + 1
    
    print(f"Found {len(words_list)} unique words from {total_occurrences} total word occurrences")
    print(f"Found {len(themes_list)} unique themes")
    print(f"Average word appears {total_occurrences / len(words_list):.2f} times")
    print(f"Most common words: {', '.join([w for w, c in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]])}")
    
    # Get embeddings in batches (OpenAI allows up to 2048 inputs per request)
    batch_size = 1024
    
    print(f"\nGetting embeddings using {embedding_model} model...")
    print(f"Note: Each unique word and theme is embedded only once, avoiding duplicates")
    
    # Track total tokens for cost calculation
    total_tokens = 0
    
    # Process words
    print(f"\nProcessing {len(words_list)} unique words...")
    word_embeddings = {}
    for i in range(0, len(words_list), batch_size):
        batch = words_list[i:i+batch_size]
        print(f"Processing word batch {i//batch_size + 1}/{(len(words_list)-1)//batch_size + 1} ({len(batch)} words)...")
        
        try:
            embeddings, tokens = get_embeddings_batch(client, batch, embedding_model)
            total_tokens += tokens
            for word, embedding in zip(batch, embeddings):
                word_embeddings[word] = embedding
        except Exception as e:
            print(f"Error processing word batch: {e}")
            return
    
    # Process themes
    print(f"\nProcessing {len(themes_list)} unique themes...")
    theme_embeddings = {}
    for i in range(0, len(themes_list), batch_size):
        batch = themes_list[i:i+batch_size]
        print(f"Processing theme batch {i//batch_size + 1}/{(len(themes_list)-1)//batch_size + 1} ({len(batch)} themes)...")
        
        try:
            embeddings, tokens = get_embeddings_batch(client, batch, embedding_model)
            total_tokens += tokens
            for theme, embedding in zip(batch, embeddings):
                theme_embeddings[theme] = embedding
        except Exception as e:
            print(f"Error processing theme batch: {e}")
            return
    
    # Create EmbeddingStore and populate it
    print(f"\nCreating EmbeddingStore...")
    store = EmbeddingStore()
    
    # Set metadata
    store.metadata = {
        "model": embedding_model,
        "created_at": datetime.now().isoformat(),
        "total_tokens_used": total_tokens
    }
    
    # Populate store with embeddings
    store.word_embeddings = word_embeddings
    store.words = sorted(list(word_embeddings.keys()))
    store.word_to_idx = {w: i for i, w in enumerate(store.words)}
    
    store.theme_embeddings = theme_embeddings
    store.themes = sorted(list(theme_embeddings.keys()))
    store.theme_to_idx = {t: i for i, t in enumerate(store.themes)}
    
    # Convert to numpy arrays for efficient operations
    try:
        import numpy as np
        store.word_vectors = np.array([word_embeddings[w] for w in store.words], dtype=np.float64)
        store.theme_vectors = np.array([theme_embeddings[t] for t in store.themes], dtype=np.float64)
        store.dimension = store.word_vectors.shape[1]
    except ImportError:
        print("Warning: numpy not installed, vectors not converted to arrays")
        store.dimension = len(next(iter(word_embeddings.values())))
    
    # Save embeddings using EmbeddingStore
    model_suffix = embedding_model.replace('/', '_')
    json_file = f"word_theme_{model_suffix}.json"
    pkl_file = f"word_theme_{model_suffix}.pkl"
    
    # Save JSON format
    print(f"Saving embeddings to {json_file}...")
    store.save_json(json_file)
    
    # Save optimized pickle format if numpy is available
    if store.word_vectors is not None:
        print(f"Creating optimized format...")
        store.save_optimized(pkl_file)
        print(f"✓ Saved optimized format to {pkl_file} ({os.path.getsize(pkl_file) / (1024*1024):.2f} MB)")
    else:
        print("Note: Install numpy to create optimized format (pip install numpy)")
    
    # Print statistics
    stats = store.stats()
    print(f"\n✓ Successfully saved embeddings for {stats['num_words']} unique words and {stats['num_themes']} unique themes")
    print(f"✓ Embedding dimensions: {stats['dimension']}")
    print(f"✓ JSON file size: {os.path.getsize(json_file) / (1024*1024):.2f} MB")
    print(f"✓ Memory usage: {stats['memory_mb']:.2f} MB")
    
    # Validate that all words and themes from examples can be looked up
    print("\n" + "="*80)
    print("VALIDATING EMBEDDING LOOKUPS")
    print("="*80)
    
    lookup_failures = {'words': [], 'themes': []}
    case_mismatches = {'words': [], 'themes': []}
    
    # Re-read examples to validate lookups
    with open("examples.jsonl", 'r') as f:
        for line_num, line in enumerate(f, 1):
            example = json.loads(line)
            
            # Check word lookups
            for word in example['words']:
                if word not in word_embeddings:
                    lookup_failures['words'].append((line_num, word))
                    # Check for case mismatch
                    for embedded_word in word_embeddings:
                        if word.lower() == embedded_word.lower():
                            case_mismatches['words'].append((line_num, word, embedded_word))
                            break
            
            # Check theme lookups
            for group in example['solution']['groups']:
                theme = group['reason']
                if theme not in theme_embeddings:
                    lookup_failures['themes'].append((line_num, theme))
                    # Check for case mismatch
                    for embedded_theme in theme_embeddings:
                        if theme.lower() == embedded_theme.lower():
                            case_mismatches['themes'].append((line_num, theme, embedded_theme))
                            break
    
    # Report validation results
    if lookup_failures['words'] or lookup_failures['themes']:
        print("\n" + "!"*80)
        print("⚠️  WARNING: EMBEDDING LOOKUP FAILURES DETECTED!")
        print("!"*80)
        
        if lookup_failures['words']:
            print(f"\n❌ {len(lookup_failures['words'])} word lookup failures:")
            for line_num, word in lookup_failures['words'][:10]:
                print(f"   Line {line_num}: '{word}' not found in embeddings")
            if len(lookup_failures['words']) > 10:
                print(f"   ... and {len(lookup_failures['words']) - 10} more")
        
        if lookup_failures['themes']:
            print(f"\n❌ {len(lookup_failures['themes'])} theme lookup failures:")
            for line_num, theme in lookup_failures['themes'][:10]:
                print(f"   Line {line_num}: '{theme}' not found in embeddings")
            if len(lookup_failures['themes']) > 10:
                print(f"   ... and {len(lookup_failures['themes']) - 10} more")
        
        if case_mismatches['words'] or case_mismatches['themes']:
            print("\n📝 Potential case mismatches detected:")
            if case_mismatches['words']:
                print(f"\n   Words with case differences:")
                for line_num, wanted, found in case_mismatches['words'][:5]:
                    print(f"     Line {line_num}: wanted '{wanted}', found '{found}'")
            if case_mismatches['themes']:
                print(f"\n   Themes with case differences:")
                for line_num, wanted, found in case_mismatches['themes'][:5]:
                    print(f"     Line {line_num}: wanted '{wanted}', found '{found}'")
        
        print("\n" + "!"*80)
        print("⚠️  FIX REQUIRED: Embeddings will not work correctly for vector operations!")
        print("!"*80)
    else:
        print("\n✅ All embedding lookups validated successfully!")
        print(f"   - All {stats['num_words']} words can be looked up")
        print(f"   - All {stats['num_themes']} themes can be looked up")
    
    # Calculate actual cost based on real token usage
    # Pricing for text-embedding-3-small: $0.02 per 1M tokens
    # Pricing for text-embedding-3-large: $0.13 per 1M tokens
    if 'small' in embedding_model:
        cost_per_million = 0.02
    elif 'large' in embedding_model:
        cost_per_million = 0.13
    else:
        cost_per_million = 0.02  # Default to small pricing
    
    actual_cost = (total_tokens / 1_000_000) * cost_per_million
    
    # Log the cost
    total_items = len(word_embeddings) + len(theme_embeddings)
    cumulative_cost = log_api_cost(actual_cost, embedding_model, total_items, "embeddings")
    
    print(f"✓ Actual tokens used: {total_tokens:,}")
    print(f"✓ API cost for this run: ${actual_cost:.4f}")
    print(f"✓ Cumulative total cost: ${cumulative_cost:.4f}")
    print(f"✓ Cost tracking saved to: openai_api_costs.txt")

if __name__ == "__main__":
    main()