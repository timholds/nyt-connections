import json
import os
from typing import List, Dict
from openai import OpenAI

def load_unique_words(filename: str) -> set:
    """Load all unique words from the examples JSONL file."""
    unique_words = set()
    
    with open(filename, 'r') as f:
        for line in f:
            example = json.loads(line)
            unique_words.update(example['words'])
    
    return unique_words

def get_embeddings_batch(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings for a batch of texts from OpenAI."""
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]

def main():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Load unique words and count occurrences
    print("Loading words from examples.jsonl...")
    unique_words = load_unique_words("examples.jsonl")
    words_list = sorted(list(unique_words))
    
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
    print(f"Average word appears {total_occurrences / len(words_list):.2f} times")
    print(f"Most common words: {', '.join([w for w, c in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]])}")
    
    # Get embeddings in batches (OpenAI allows up to 2048 inputs per request)
    batch_size = 100
    embeddings_dict = {}
    
    print(f"\nGetting embeddings using text-embedding-3-small model...")
    print(f"Note: Each unique word is embedded only once, avoiding duplicates")
    
    for i in range(0, len(words_list), batch_size):
        batch = words_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(words_list)-1)//batch_size + 1} ({len(batch)} words)...")
        
        try:
            embeddings = get_embeddings_batch(client, batch)
            for word, embedding in zip(batch, embeddings):
                embeddings_dict[word] = embedding
        except Exception as e:
            print(f"Error processing batch: {e}")
            return
    
    # Save embeddings
    output_file = "word_embeddings.json"
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f)
    
    # Print statistics
    print(f"\n✓ Successfully saved embeddings for {len(embeddings_dict)} unique words")
    print(f"✓ Embedding dimensions: {len(next(iter(embeddings_dict.values())))}")
    print(f"✓ Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Calculate estimated cost (more accurate)
    # OpenAI charges per token, roughly 4 chars per token
    total_chars = sum(len(word) for word in words_list)
    total_tokens = total_chars / 4  # Rough estimate
    estimated_cost = (total_tokens / 1_000_000) * 0.02
    print(f"✓ Estimated API cost: ~${estimated_cost:.4f}")

if __name__ == "__main__":
    main()