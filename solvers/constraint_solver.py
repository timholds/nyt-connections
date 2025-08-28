import os
from typing import List, Dict, Any, Optional, Tuple, Set
import itertools
from dataclasses import dataclass
import numpy as np
from .base import BaseSolver
from .models import PuzzleSolution, GroupSolution
from openai import OpenAI
import json


@dataclass
class CandidateGroup:
    """Represents a candidate group that may overlap with others."""
    words: List[str]
    reason: str
    confidence: float = 0.0
    
    def __hash__(self):
        return hash(tuple(sorted(self.words)))
    
    def __eq__(self, other):
        return set(self.words) == set(other.words)


class ConstraintValidationSolver(BaseSolver):
    """
    Solver using LLM Chain-of-Thought with Constraint Validation.
    
    Key innovation: Generate many overlapping candidate groups, then use
    constraint solving to find the optimal valid partition.
    
    Process:
    1. Generate 8-12 candidate groups (may have 3-5 words, may overlap)
    2. Use constraint solver to find valid 4x4 partition
    3. Re-prompt if no valid solution found
    """
    
    def __init__(self, examples: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.examples = examples or []
        
    def get_system_prompt(self, current_words: Optional[List[str]] = None) -> str:
        """Get system prompt for candidate generation."""
        return """You are an expert at solving NYT Connections puzzles. 
        
Your task is to generate MULTIPLE POSSIBLE GROUPINGS for the given words.
Important: These groups MAY OVERLAP - the same word can appear in multiple groups.
This allows you to express uncertainty and multiple interpretations.

Focus on different interpretation strategies:
- Literal categories (animals, colors, tools, etc.)
- Wordplay (homophones, rhymes, word parts)
- Common phrases or idioms
- Pop culture references
- Double meanings of words

Generate 8-12 candidate groups. Each group should have 3-5 words."""

    def get_user_message(self, words: List[str]) -> str:
        """Format user message for candidate generation."""
        return f"""Find multiple possible groupings for these 16 words:
{', '.join(words)}

Generate 8-12 candidate groups. Groups MAY overlap (same word in multiple groups).
Each group should have 3-5 words that share a clear connection.

Output format (JSON):
{{
  "candidates": [
    {{
      "words": ["word1", "word2", "word3", "word4"],
      "reason": "Things that are blue",
      "confidence": 0.9
    }},
    {{
      "words": ["word2", "word5", "word6"],
      "reason": "Can follow 'break'",
      "confidence": 0.7
    }}
  ]
}}

Include confidence scores (0-1) for each group."""

    def parse_candidate_response(self, response: str) -> List[CandidateGroup]:
        """Parse LLM response into candidate groups."""
        try:
            data = json.loads(response)
            candidates = []
            for item in data.get('candidates', []):
                candidates.append(CandidateGroup(
                    words=item['words'],
                    reason=item['reason'],
                    confidence=item.get('confidence', 0.5)
                ))
            return candidates
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse JSON response: {e}")
            # Fallback: try to extract groups from text
            return self.parse_text_response(response)
    
    def parse_text_response(self, response: str) -> List[CandidateGroup]:
        """Fallback parser for non-JSON responses."""
        candidates = []
        lines = response.split('\n')
        
        for i, line in enumerate(lines):
            # Look for various patterns
            line = line.strip()
            
            # Pattern 1: Group N: [words] - reason
            # Pattern 2: N. [words] - reason
            # Pattern 3: - [words]: reason
            
            words_list = []
            reason = ""
            confidence = 0.5
            
            # Try to find words in various formats
            if '[' in line and ']' in line:
                try:
                    start = line.index('[')
                    end = line.index(']')
                    words_str = line[start+1:end]
                    words_list = [w.strip().strip('"\'') for w in words_str.split(',')]
                    
                    # Extract reason after brackets
                    reason_part = line[end+1:].strip()
                    if '-' in reason_part:
                        reason = reason_part.split('-', 1)[-1].strip()
                    elif ':' in reason_part:
                        reason = reason_part.split(':', 1)[-1].strip()
                    else:
                        reason = reason_part
                except:
                    pass
            
            # Alternative: Look for quoted word lists
            elif '"' in line or "'" in line:
                try:
                    import re
                    # Find all quoted strings
                    quoted = re.findall(r'["\']([^"\']+)["\']', line)
                    if len(quoted) >= 3:  # At least 3 words for a group
                        words_list = quoted[:5]  # Take up to 5
                        # Try to find reason after the words
                        last_quote_pos = max(line.rfind('"'), line.rfind("'"))
                        if last_quote_pos > 0:
                            reason_part = line[last_quote_pos+1:].strip()
                            if ':' in reason_part:
                                reason = reason_part.split(':', 1)[-1].strip()
                            elif '-' in reason_part:
                                reason = reason_part.split('-', 1)[-1].strip()
                            else:
                                reason = reason_part
                except:
                    pass
            
            # Look for confidence scores
            if 'confidence' in line.lower() or 'conf:' in line.lower():
                try:
                    import re
                    conf_match = re.search(r'(?:confidence|conf)[:\s]*([0-9.]+)', line.lower())
                    if conf_match:
                        confidence = float(conf_match.group(1))
                except:
                    pass
            
            if words_list and len(words_list) >= 3:
                candidates.append(CandidateGroup(
                    words=words_list,
                    reason=reason if reason else f"Group {len(candidates)+1}",
                    confidence=confidence
                ))
        
        return candidates

    def generate_candidates(self, words: List[str], use_api: bool, model: str) -> List[CandidateGroup]:
        """Generate candidate groups using LLM."""
        if not use_api:
            # Dummy candidates for testing
            return [
                CandidateGroup(words[:4], "Test group 1", 0.8),
                CandidateGroup(words[2:6], "Test group 2", 0.7),
                CandidateGroup(words[4:8], "Test group 3", 0.6),
                CandidateGroup(words[8:12], "Test group 4", 0.9),
                CandidateGroup(words[12:16], "Test group 5", 0.75),
            ]
        
        # Initialize client if needed
        if not self.client:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Make API call for candidate generation
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.get_system_prompt(words)},
                {"role": "user", "content": self.get_user_message(words)}
            ],
            temperature=0.8,  # Higher temp for more diverse candidates
            max_tokens=1500
        )
        
        # Track API costs
        usage = response.usage
        pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["gpt-4o-mini"])
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        self.log_api_cost(model, usage.prompt_tokens, usage.completion_tokens, total_cost)
        print(f"Candidate generation cost: ${total_cost:.4f}")
        
        # Parse response
        content = response.choices[0].message.content
        candidates = self.parse_candidate_response(content)
        
        print(f"Generated {len(candidates)} candidate groups")
        return candidates

    def find_valid_partition_greedy(self, candidates: List[CandidateGroup], words: List[str]) -> Optional[List[CandidateGroup]]:
        """
        Greedy algorithm to find a valid partition.
        Prioritizes high-confidence groups with exactly 4 words.
        """
        # Filter to groups with exactly 4 words
        valid_candidates = [c for c in candidates if len(c.words) == 4]
        
        # Sort by confidence (highest first)
        valid_candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        # Try to find 4 non-overlapping groups
        selected = []
        used_words = set()
        
        for candidate in valid_candidates:
            # Check if this group overlaps with already selected groups
            candidate_words = set(candidate.words)
            if not candidate_words & used_words:  # No overlap
                selected.append(candidate)
                used_words.update(candidate_words)
                
                if len(selected) == 4:
                    # Check if we've used all 16 words
                    if len(used_words) == 16:
                        return selected
                    else:
                        # Continue searching for better combination
                        selected.pop()
                        used_words -= candidate_words
        
        return None

    def find_valid_partition_exhaustive(self, candidates: List[CandidateGroup], words: List[str], max_combinations: int = 1000) -> Optional[List[CandidateGroup]]:
        """
        Exhaustive search for valid partition (with limit).
        Tries combinations of 4 groups from candidates.
        """
        # Filter to groups with exactly 4 words
        valid_candidates = [c for c in candidates if len(c.words) == 4]
        
        if len(valid_candidates) < 4:
            return None
        
        # Try combinations of 4 groups
        word_set = set(words)
        best_partition = None
        best_score = -1
        
        for i, combination in enumerate(itertools.combinations(valid_candidates, 4)):
            if i >= max_combinations:
                break
                
            # Check if this is a valid partition
            all_words_in_combo = set()
            overlap = False
            
            for group in combination:
                group_words = set(group.words)
                if all_words_in_combo & group_words:  # Check overlap
                    overlap = True
                    break
                all_words_in_combo.update(group_words)
            
            # Valid partition must cover all 16 words with no overlap
            if not overlap and all_words_in_combo == word_set:
                # Score this partition
                score = sum(g.confidence for g in combination)
                if score > best_score:
                    best_score = score
                    best_partition = list(combination)
        
        return best_partition

    def expand_candidates(self, candidates: List[CandidateGroup], words: List[str]) -> List[CandidateGroup]:
        """
        Expand partial candidates (3 or 5 words) to exactly 4 words.
        """
        expanded = []
        used_words_count = {}
        
        # Count word usage across all candidates
        for c in candidates:
            for word in c.words:
                used_words_count[word] = used_words_count.get(word, 0) + 1
        
        for candidate in candidates:
            if len(candidate.words) == 4:
                expanded.append(candidate)
            elif len(candidate.words) == 3:
                # Find best word to add
                unused_in_group = [w for w in words if w not in candidate.words]
                if unused_in_group:
                    # Prefer less-used words
                    unused_in_group.sort(key=lambda w: used_words_count.get(w, 0))
                    new_words = candidate.words + [unused_in_group[0]]
                    expanded.append(CandidateGroup(
                        words=new_words,
                        reason=candidate.reason + " (expanded)",
                        confidence=candidate.confidence * 0.9
                    ))
            elif len(candidate.words) == 5:
                # Try removing each word and keep best group of 4
                for i in range(5):
                    new_words = candidate.words[:i] + candidate.words[i+1:]
                    expanded.append(CandidateGroup(
                        words=new_words,
                        reason=candidate.reason + " (reduced)",
                        confidence=candidate.confidence * 0.9
                    ))
        
        return expanded

    def solve(self, words: List[str], use_api: bool = False, model: str = "gpt-4o-mini") -> PuzzleSolution:
        """
        Solve using constraint validation approach.
        """
        # Validate input
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        if len(set(words)) != 16:
            raise ValueError("All 16 words must be unique")

        print("\n" + "="*60)
        print("CONSTRAINT VALIDATION SOLVER")
        print("="*60)
        
        # Phase 1: Generate overlapping candidates
        print("\n📝 Phase 1: Generating candidate groups (may overlap)...")
        candidates = self.generate_candidates(words, use_api, model)
        
        if use_api and candidates:
            # Display candidates
            print(f"\nGenerated {len(candidates)} candidates:")
            for i, c in enumerate(candidates[:8], 1):  # Show first 8
                print(f"  {i}. [{', '.join(c.words[:4])}{'...' if len(c.words) > 4 else ''}] - {c.reason} (conf: {c.confidence:.2f})")
            if len(candidates) > 8:
                print(f"  ... and {len(candidates) - 8} more")
        
        # Expand partial candidates to exactly 4 words
        print("\n🔄 Expanding partial candidates to 4 words...")
        expanded_candidates = self.expand_candidates(candidates, words)
        print(f"Expanded to {len(expanded_candidates)} candidates")
        
        # Phase 2: Find valid partition using constraint solver
        print("\n🔍 Phase 2: Finding valid partition (no overlaps, all words used)...")
        
        # Try greedy first (fast)
        solution = self.find_valid_partition_greedy(expanded_candidates, words)
        
        if not solution:
            print("  Greedy failed, trying exhaustive search...")
            solution = self.find_valid_partition_exhaustive(expanded_candidates, words)
        
        if solution:
            print(f"✅ Found valid partition with combined confidence: {sum(g.confidence for g in solution):.2f}")
        else:
            print("❌ No valid partition found, using fallback")
            # Fallback: use top 4 non-overlapping groups or create dummy groups
            solution = self.create_fallback_solution(expanded_candidates, words)
        
        # Convert to PuzzleSolution format
        groups = []
        for group in solution:
            groups.append(GroupSolution(
                words=group.words,
                reason=group.reason
            ))
        
        return PuzzleSolution(groups=groups)

    def create_fallback_solution(self, candidates: List[CandidateGroup], words: List[str]) -> List[CandidateGroup]:
        """Create a fallback solution when no valid partition is found."""
        # Try to use as many high-confidence candidates as possible
        selected = []
        used_words = set()
        
        # Sort by confidence
        candidates_4 = [c for c in candidates if len(c.words) == 4]
        candidates_4.sort(key=lambda c: c.confidence, reverse=True)
        
        for candidate in candidates_4:
            # Use groups that don't overlap
            candidate_words = set(candidate.words)
            overlap = candidate_words & used_words
            
            if not overlap:  # No overlap at all
                selected.append(candidate)
                used_words.update(candidate_words)
                
                if len(selected) == 4:
                    # Check if we've covered all 16 words
                    if len(used_words) == 16:
                        return selected
                    else:
                        # Not all words covered, continue
                        selected.pop()
                        used_words -= candidate_words
        
        # If we couldn't find 4 non-overlapping groups, create simple fallback
        if len(selected) < 4:
            # Reset and create simple division
            selected = []
            remaining_words = list(words)
            for i in range(4):
                group_words = remaining_words[i*4:(i+1)*4]
                selected.append(CandidateGroup(
                    words=group_words,
                    reason=f"Fallback group {i+1}",
                    confidence=0.1
                ))
        
        return selected