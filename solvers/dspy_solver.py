import os
import json
from typing import List, Dict, Any, Optional
import dspy
from .base import BaseSolver
from .models import PuzzleSolution, GroupSolution


class ConnectionsSignature(dspy.Signature):
    """Solve a NYT Connections puzzle by grouping 16 words into 4 categories."""
    
    words = dspy.InputField(desc="16 words to group into 4 categories")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning exploring different grouping strategies")
    group1_words = dspy.OutputField(desc="First group of exactly 4 words (comma-separated)")
    group1_reason = dspy.OutputField(desc="Connecting theme for group 1")
    group2_words = dspy.OutputField(desc="Second group of exactly 4 words (comma-separated)")
    group2_reason = dspy.OutputField(desc="Connecting theme for group 2")
    group3_words = dspy.OutputField(desc="Third group of exactly 4 words (comma-separated)")
    group3_reason = dspy.OutputField(desc="Connecting theme for group 3")
    group4_words = dspy.OutputField(desc="Fourth group of exactly 4 words (comma-separated)")
    group4_reason = dspy.OutputField(desc="Connecting theme for group 4")


class ConnectionsSolver(dspy.Module):
    """DSPy module for solving Connections puzzles."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ConnectionsSignature)
    
    def forward(self, words: str):
        """Forward pass with constraint validation."""
        
        # Generate solution with reasoning
        pred = self.generate(words=words)
        
        # Parse groups from the prediction
        all_groups = []
        all_words_used = []
        
        for i in range(1, 5):
            group_words = getattr(pred, f"group{i}_words", "")
            # Clean and split the words
            words_list = [w.strip().lower() for w in group_words.split(",")]
            all_groups.append(words_list)
            all_words_used.extend(words_list)
        
        # Basic validation
        input_words = set(w.strip().lower() for w in words.split(","))
        
        # Check constraints
        issues = []
        for i, group in enumerate(all_groups, 1):
            if len(group) != 4:
                issues.append(f"Group {i} has {len(group)} words instead of 4")
        
        if len(all_words_used) != len(set(all_words_used)):
            duplicates = [w for w in all_words_used if all_words_used.count(w) > 1]
            issues.append(f"Duplicate words found: {set(duplicates)}")
        
        if set(all_words_used) != input_words:
            missing = input_words - set(all_words_used)
            extra = set(all_words_used) - input_words
            if missing:
                issues.append(f"Missing words: {missing}")
            if extra:
                issues.append(f"Extra words: {extra}")
        
        if issues:
            print(f"Warning - Constraint violations: {'; '.join(issues)}")
        
        return pred


class DSPySolver(BaseSolver):
    """DSPy-based solver using OPTIMIZED few-shot learning and chain of thought.
    
    This solver uses DSPy's capabilities for:
    - Dynamic example selection based on similarity
    - Chain-of-thought reasoning
    - Potential for example optimization (vs naive static examples in FewShotSolver)
    """
    
    def __init__(self, examples: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.examples = examples or []
        self.dspy_solver = None
        self.lm = None
        
    
    def get_similar_examples(self, words: List[str], n: int = 3) -> List[Dict]:
        """Get n most similar examples based on word/theme patterns."""
        if not self.examples:
            return []
        
        # Score examples by theme similarity patterns
        scores = []
        word_set = set(w.lower() for w in words)
        
        for example in self.examples:
            example_words = set(w.lower() for w in example['words'])
            
            # Score based on:
            # 1. Similar word patterns (e.g., short words, capitalized words)
            # 2. Theme diversity in the example
            score = 0
            
            # Check for similar word lengths
            avg_len_input = sum(len(w) for w in words) / len(words)
            avg_len_example = sum(len(w) for w in example['words']) / len(example['words'])
            if abs(avg_len_input - avg_len_example) < 2:
                score += 1
            
            # Check for similar capitalization patterns
            caps_input = sum(1 for w in words if w[0].isupper())
            caps_example = sum(1 for w in example['words'] if w[0].isupper())
            if abs(caps_input - caps_example) < 3:
                score += 1
            
            # Avoid exact matches
            if word_set == example_words:
                score = -100
            
            scores.append((score, example))
        
        # Get top n examples
        scores.sort(key=lambda x: x[0], reverse=True)
        # Filter out exact matches (score = -100) to avoid giving away answers
        return [ex for score, ex in scores[:n] if score > -100]
    
    def format_dspy_example(self, example: Dict) -> dspy.Example:
        """Format an example for DSPy."""
        words_str = ", ".join(example['words'])
        
        # Build reasoning that explores different strategies
        groups = example['solution']['groups']
        reasoning_parts = [
            "Let me analyze these words systematically:",
            "",
            "First, I'll look for obvious categories and literal meanings:",
        ]
        
        # Add some example reasoning
        if groups:
            reasoning_parts.append(f"- {', '.join(groups[0]['words'][:2])}... these might be {groups[0]['reason']}")
        
        reasoning_parts.extend([
            "",
            "Next, checking for wordplay patterns (homophones, palindromes, rhymes):",
            "- Analyzing word structures and sounds...",
            "",
            "Looking for phrases and idioms:",
            "- Words that commonly go together or complete phrases...",
            "",
            "Considering pop culture and specialized knowledge:",
            "- Brand names, geography, entertainment references...",
            "",
            "After careful analysis, I've identified these four groups:"
        ])
        
        reasoning = "\n".join(reasoning_parts)
        
        # Create the example with all fields
        return dspy.Example(
            words=words_str,
            reasoning=reasoning,
            group1_words=", ".join(groups[0]['words']),
            group1_reason=groups[0]['reason'],
            group2_words=", ".join(groups[1]['words']),
            group2_reason=groups[1]['reason'],
            group3_words=", ".join(groups[2]['words']),
            group3_reason=groups[2]['reason'],
            group4_words=", ".join(groups[3]['words']),
            group4_reason=groups[3]['reason']
        ).with_inputs('words')
    
    def get_system_prompt(self, current_words: Optional[List[str]] = None) -> str:
        """Get the system prompt for DSPy (not used directly but kept for compatibility)."""
        return "You are an expert at solving NYT Connections puzzles."
    
    def get_user_message(self, words: List[str]) -> str:
        """Format user message (not used directly in DSPy but kept for compatibility)."""
        return ", ".join(words)
    
    def solve(self, words: List[str], use_api: bool = False, model: str = "gpt-4o-mini") -> PuzzleSolution:
        """
        Solve using DSPy with few-shot examples and chain of thought.
        
        Args:
            words: List of 16 words to group
            use_api: Whether to make actual API call
            model: OpenAI model to use
        
        Returns:
            PuzzleSolution with structured groups
        """
        # Validate input
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        if len(set(words)) != 16:
            raise ValueError("All 16 words must be unique")
        
        if use_api:
            # Initialize DSPy with the specified model
            if not self.lm:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                
                self.lm = dspy.LM(
                    model=model,
                    api_key=api_key,
                    max_tokens=800,
                    temperature=0.7
                )
                dspy.settings.configure(lm=self.lm)
            
            # Use provided examples
            if self.examples:
                print(f"Using {len(self.examples)} training examples")
            
            # Initialize solver if needed
            if not self.dspy_solver:
                self.dspy_solver = ConnectionsSolver()
            
            # Get similar examples for few-shot learning
            if self.examples:
                similar = self.get_similar_examples(words, n=3)
                if similar:
                    print(f"Using {len(similar)} similar examples for few-shot context")
                    
                    # Format examples for few-shot prompting
                    formatted_examples = []
                    for ex in similar[:2]:  # Use only 2 to avoid context length issues
                        formatted = self.format_dspy_example(ex)
                        formatted_examples.append(formatted)
                    
                    # Set examples in DSPy context
                    self.dspy_solver.generate.demos = formatted_examples
            
            # Solve the puzzle
            words_str = ", ".join(words)
            
            try:
                # Try to solve with retries
                max_retries = 3
                best_pred = None
                validation_issues = []
                
                for attempt in range(max_retries):
                    try:
                        # Add retry feedback if we had validation issues
                        if attempt > 0 and validation_issues:
                            # Temporarily modify the prompt with feedback
                            original_demos = self.dspy_solver.generate.demos
                            retry_feedback = dspy.Example(
                                words=words_str,
                                reasoning=f"IMPORTANT: Previous attempt had issues: {', '.join(validation_issues)}. Ensure each group has EXACTLY 4 words and all 16 words are used exactly once.",
                                group1_words="", group1_reason="",
                                group2_words="", group2_reason="",
                                group3_words="", group3_reason="",
                                group4_words="", group4_reason=""
                            ).with_inputs('words', 'reasoning')
                            
                            # Add feedback as a demo temporarily
                            self.dspy_solver.generate.demos = [retry_feedback] + (original_demos or [])
                        
                        pred = self.dspy_solver(words=words_str)
                        
                        # Validate the prediction
                        validation_issues = []
                        all_words_used = []
                        
                        for i in range(1, 5):
                            group_words = getattr(pred, f"group{i}_words", "")
                            words_list = [w.strip() for w in group_words.split(",") if w.strip()]
                            
                            if len(words_list) != 4:
                                validation_issues.append(f"Group {i} has {len(words_list)} words instead of 4")
                            
                            all_words_used.extend(words_list)
                        
                        # Check for duplicates
                        if len(all_words_used) != len(set(all_words_used)):
                            duplicates = [w for w in set(all_words_used) if all_words_used.count(w) > 1]
                            validation_issues.append(f"Duplicate words: {duplicates}")
                        
                        # Check if all 16 words are used
                        if len(set(all_words_used)) != 16:
                            validation_issues.append(f"Used {len(set(all_words_used))}/16 unique words")
                        
                        if not validation_issues or attempt == max_retries - 1:
                            best_pred = pred
                            if validation_issues:
                                print(f"Warning: Using solution with issues after {max_retries} attempts")
                            break
                        else:
                            print(f"Attempt {attempt + 1}/{max_retries} validation issues: {validation_issues}")
                            # Adjust temperature slightly for variety
                            self.lm.kwargs['temperature'] = min(0.9, self.lm.kwargs.get('temperature', 0.7) + 0.05)
                            
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed with error: {e}")
                        if attempt == max_retries - 1:
                            raise
                    finally:
                        # Restore original demos if we modified them
                        if attempt > 0 and validation_issues:
                            self.dspy_solver.generate.demos = original_demos
                
                pred = best_pred
                
                # Parse the DSPy output into our PuzzleSolution format
                groups = []
                for i in range(1, 5):
                    group_words = getattr(pred, f"group{i}_words", "")
                    group_reason = getattr(pred, f"group{i}_reason", "")
                    
                    # Clean and parse words
                    words_list = [w.strip() for w in group_words.split(",")]
                    
                    # Ensure we have exactly 4 words
                    if len(words_list) > 4:
                        words_list = words_list[:4]
                    elif len(words_list) < 4:
                        # Try to fill with remaining words not used yet
                        used_words = set()
                        for j in range(1, i):
                            prev_words = getattr(pred, f"group{j}_words", "")
                            used_words.update(w.strip() for w in prev_words.split(","))
                        
                        remaining = [w for w in words if w not in used_words]
                        words_list.extend(remaining[:4-len(words_list)])
                    
                    groups.append(GroupSolution(
                        words=words_list,
                        reason=group_reason or f"Group {i}"
                    ))
                
                solution = PuzzleSolution(groups=groups)
                
                # Print reasoning if available
                if hasattr(pred, 'reasoning') and pred.reasoning:
                    print("\nDSPy Chain-of-Thought Reasoning:")
                    print("-" * 40)
                    print(pred.reasoning[:500] + "..." if len(pred.reasoning) > 500 else pred.reasoning)
                    print("-" * 40)
                
            except Exception as e:
                print(f"DSPy solving failed: {e}")
                # Fallback to a simple solution
                solution = PuzzleSolution(
                    groups=[
                        GroupSolution(words=list(words[:4]), reason="fallback group 1"),
                        GroupSolution(words=list(words[4:8]), reason="fallback group 2"),
                        GroupSolution(words=list(words[8:12]), reason="fallback group 3"),
                        GroupSolution(words=list(words[12:16]), reason="fallback group 4")
                    ]
                )
            
            return solution
            
        else:
            # Dummy response for testing
            print("=" * 80)
            print("DSPy Solver - Test Mode")
            print("Would use Chain-of-Thought with few-shot examples")
            print("Features: Dynamic example selection, multi-strategy reasoning, constraint validation")
            print("=" * 80)
            
            return PuzzleSolution(
                groups=[
                    GroupSolution(words=list(words[:4]), reason="dummy group 1"),
                    GroupSolution(words=list(words[4:8]), reason="dummy group 2"),
                    GroupSolution(words=list(words[8:12]), reason="dummy group 3"),
                    GroupSolution(words=list(words[12:16]), reason="dummy group 4")
                ]
            )