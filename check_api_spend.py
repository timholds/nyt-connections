#!/usr/bin/env python3
"""Check current OpenAI API spending from the tracking file"""
import os
import sys

def check_spending():
    log_file = "openai_api_costs.txt"
    
    if not os.path.exists(log_file):
        print("No API costs tracked yet.")
        print("Run get_embeddings.py to start tracking costs.")
        return
    
    # Read the file and find the last TOTAL line
    total_cost = 0
    entries = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    current_entry = []
    for line in lines:
        if line.startswith("-" * 10):
            if current_entry:
                entries.append(current_entry)
                current_entry = []
        else:
            current_entry.append(line.strip())
            if line.startswith("TOTAL:"):
                total_cost = float(line.split('$')[1])
    
    # Print summary
    print("=" * 60)
    print("OpenAI API Spending Summary")
    print("=" * 60)
    print(f"\nTotal spending: ${total_cost:.4f}")
    print(f"Number of API calls tracked: {len(entries)}")
    
    if entries:
        print("\nLast 5 entries:")
        print("-" * 60)
        for entry in entries[-5:]:
            for line in entry:
                if not line.startswith("TOTAL:"):
                    print(line)
    
    print("\nFor full details, see: openai_api_costs.txt")

if __name__ == "__main__":
    check_spending()