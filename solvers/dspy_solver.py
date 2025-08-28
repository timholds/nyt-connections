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
    
    def __init__(self, examples: Optional[List[Dict[str, Any]]] = None, use_optimized: bool = True):
        super().__init__()
        self.examples = examples or []
        self.dspy_solver = None
        self.lm = None
        self.use_optimized = use_optimized
        
    
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
    
    def _create_smart_fallback(self, words: List[str]) -> PuzzleSolution:
        """Create a smarter fallback solution using basic heuristics."""
        print("Creating smart fallback solution...")
        
        # Try some basic groupings based on word patterns
        groups = []
        used_words = set()
        
        # Group 1: Try to find short words (likely common category)
        short_words = [w for w in words if len(w) <= 4 and w.lower() not in used_words]
        if len(short_words) >= 4:
            group1 = short_words[:4]
            used_words.update(w.lower() for w in group1)
            groups.append(GroupSolution(words=group1, reason="Short words"))
        
        # Group 2: Try to find capitalized words (likely proper nouns)
        remaining = [w for w in words if w.lower() not in used_words]
        cap_words = [w for w in remaining if w[0].isupper() and len(w) > 1]
        if len(cap_words) >= 4:
            group2 = cap_words[:4]
            used_words.update(w.lower() for w in group2)
            groups.append(GroupSolution(words=group2, reason="Capitalized words"))
        
        # Group 3: Try to find words with similar lengths
        remaining = [w for w in words if w.lower() not in used_words]
        if remaining:
            # Group by length
            by_length = {}
            for w in remaining:
                length = len(w)
                if length not in by_length:
                    by_length[length] = []
                by_length[length].append(w)
            
            # Find the length with most words
            best_length = max(by_length.keys(), key=lambda l: len(by_length[l]))
            if len(by_length[best_length]) >= 4:
                group3 = by_length[best_length][:4]
                used_words.update(w.lower() for w in group3)
                groups.append(GroupSolution(words=group3, reason=f"Words of length {best_length}"))
        
        # Fill remaining groups with leftover words
        remaining = [w for w in words if w.lower() not in used_words]
        group_num = len(groups) + 1
        
        while len(groups) < 4 and remaining:
            group_size = min(4, len(remaining))
            if len(remaining) > 4 and len(groups) == 2:
                # Split remaining words as evenly as possible between last 2 groups
                mid = len(remaining) // 2
                group_words = remaining[:mid] if len(remaining) - mid >= 4 else remaining[:4]
            else:
                group_words = remaining[:group_size]
            
            groups.append(GroupSolution(
                words=group_words, 
                reason=f"Fallback group {group_num}"
            ))
            
            used_words.update(w.lower() for w in group_words)
            remaining = [w for w in remaining if w.lower() not in used_words]
            group_num += 1
        
        # If we don't have exactly 4 groups, fall back to simple split
        if len(groups) != 4:
            print("Smart fallback failed, using simple split")
            groups = [
                GroupSolution(words=list(words[:4]), reason="fallback group 1"),
                GroupSolution(words=list(words[4:8]), reason="fallback group 2"),
                GroupSolution(words=list(words[8:12]), reason="fallback group 3"),
                GroupSolution(words=list(words[12:16]), reason="fallback group 4")
            ]
        
        return PuzzleSolution(groups=groups)

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
    
    def solve(self, words: List[str], use_api: bool = False, model: str = "gpt-5-mini") -> PuzzleSolution:
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
            
            # Initialize solver if needed
            if not self.dspy_solver:
                # Try model-specific optimization file first, then generic
                model_suffix = model.replace("-", "_")
                optimized_files = [
                    f'optimized_solver_{model_suffix}.json',
                    'optimized_solver.json'
                ]
                
                loaded_optimization = False
                for opt_file in optimized_files:
                    if self.use_optimized and os.path.exists(opt_file):
                        # Load the MIPRO-optimized module
                        print(f"Loading MIPRO-optimized solver configuration from {opt_file}...")
                        self.dspy_solver = dspy.Module()
                        self.dspy_solver.load(opt_file)
                        loaded_optimization = True
                        
                        # If the loaded module doesn't have a generate method, wrap it
                        if not hasattr(self.dspy_solver, 'generate'):
                            # Create a ChainOfThought with the optimized signature
                            with open(opt_file, 'r') as f:
                                config = json.load(f)
                        
                            # The optimized config contains the signature and demos
                            if 'generate.predict' in config:
                                opt_config = config['generate.predict']
                                
                                # Create a custom signature with optimized instructions
                                class OptimizedConnectionsSignature(dspy.Signature):
                                    __doc__ = opt_config['signature']['instructions']
                                    
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
                                
                                # Create a solver with the optimized signature
                                solver = ConnectionsSolver()
                                solver.generate = dspy.ChainOfThought(OptimizedConnectionsSignature)
                                
                                # Apply any optimized demos if they exist
                                if 'demos' in opt_config and opt_config['demos']:
                                    solver.generate.demos = opt_config['demos']
                                
                                self.dspy_solver = solver
                            else:
                                # Fallback to standard solver
                                print("Warning: Optimized config format not recognized, using standard solver")
                                self.dspy_solver = ConnectionsSolver()
                        break  # Successfully loaded optimization
                
                # If no optimization was loaded, use standard solver
                if not loaded_optimization:
                    self.dspy_solver = ConnectionsSolver()
            
            # Get similar examples for few-shot learning
            if self.examples:
                similar = self.get_similar_examples(words, n=3)
                if similar:
                    # We actually only use 2 examples to avoid context length issues
                    examples_used = min(2, len(similar))
                    print(f"Selected {examples_used} relevant examples from {len(self.examples)} available training examples")
                    
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
                        # Store original demos before any modifications
                        original_demos = getattr(self.dspy_solver.generate, 'demos', None)
                        
                        # Add retry feedback if we had validation issues
                        if attempt > 0 and validation_issues:
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
                        input_words_set = set(w.lower().strip() for w in words)
                        
                        for i in range(1, 5):
                            group_words = getattr(pred, f"group{i}_words", "")
                            # More robust word parsing - handle different separators and case
                            words_list = []
                            for w in group_words.replace(';', ',').replace('|', ',').split(","):
                                w = w.strip().lower()
                                if w:
                                    words_list.append(w)
                            
                            if len(words_list) != 4:
                                validation_issues.append(f"Group {i} has {len(words_list)} words instead of 4")
                            
                            # Check if words are from the original input
                            for word in words_list:
                                if word not in input_words_set:
                                    validation_issues.append(f"Group {i} contains '{word}' which is not in the input")
                            
                            all_words_used.extend(words_list)
                        
                        # Check for duplicates
                        if len(all_words_used) != len(set(all_words_used)):
                            duplicates = [w for w in all_words_used if all_words_used.count(w) > 1]
                            validation_issues.append(f"Duplicate words: {duplicates}")
                        
                        # Check if all 16 words are used exactly once
                        used_set = set(all_words_used)
                        if used_set != input_words_set:
                            missing = input_words_set - used_set
                            extra = used_set - input_words_set
                            if missing:
                                validation_issues.append(f"Missing words: {list(missing)}")
                            if extra:
                                validation_issues.append(f"Extra words: {list(extra)}")
                        
                        if len(all_words_used) != 16:
                            validation_issues.append(f"Used {len(all_words_used)} total words instead of 16")
                        
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
                            if original_demos is not None:
                                self.dspy_solver.generate.demos = original_demos
                            else:
                                # Clear demos if there were none originally
                                if hasattr(self.dspy_solver.generate, 'demos'):
                                    self.dspy_solver.generate.demos = None
                
                pred = best_pred
                
                # Track API costs from DSPy
                if hasattr(self.lm, '_total_cost') and self.lm._total_cost > 0:
                    # DSPy tracks costs internally, log it
                    self.log_api_cost(model, 0, 0, self.lm._total_cost)
                    print(f"DSPy API cost: ${self.lm._total_cost:.4f}")
                elif hasattr(self.lm, 'history') and self.lm.history:
                    # Calculate cost from history
                    total_cost = 0
                    total_prompt_tokens = 0
                    total_completion_tokens = 0
                    
                    for entry in self.lm.history:
                        if hasattr(entry, 'usage') and entry.usage:
                            prompt_tokens = entry.usage.prompt_tokens
                            completion_tokens = entry.usage.completion_tokens
                            total_prompt_tokens += prompt_tokens
                            total_completion_tokens += completion_tokens
                            
                            pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["gpt-5-mini"])
                            cost = (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]
                            total_cost += cost
                    
                    if total_cost > 0:
                        self.log_api_cost(model, total_prompt_tokens, total_completion_tokens, total_cost)
                        print(f"DSPy API cost: ${total_cost:.4f}")
                
                # Parse the DSPy output into our PuzzleSolution format
                groups = []
                all_used_words = set()
                input_words_lower = {w.lower(): w for w in words}  # map lowercase to original case
                
                for i in range(1, 5):
                    group_words = getattr(pred, f"group{i}_words", "")
                    group_reason = getattr(pred, f"group{i}_reason", "")
                    
                    # Clean and parse words with robust handling
                    words_list = []
                    for w in group_words.replace(';', ',').replace('|', ',').split(","):
                        w = w.strip()
                        if w:
                            # Try to match original word case
                            w_lower = w.lower()
                            if w_lower in input_words_lower:
                                words_list.append(input_words_lower[w_lower])
                            else:
                                # If we can't match, keep the original
                                words_list.append(w)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_words = []
                    for w in words_list:
                        w_lower = w.lower()
                        if w_lower not in seen and w_lower not in all_used_words:
                            unique_words.append(w)
                            seen.add(w_lower)
                            all_used_words.add(w_lower)
                    
                    # Ensure we have exactly 4 words
                    if len(unique_words) > 4:
                        unique_words = unique_words[:4]
                    elif len(unique_words) < 4:
                        # Fill with remaining unused words
                        remaining = [w for w in words if w.lower() not in all_used_words]
                        needed = 4 - len(unique_words)
                        unique_words.extend(remaining[:needed])
                        all_used_words.update(w.lower() for w in remaining[:needed])
                    
                    groups.append(GroupSolution(
                        words=unique_words,
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
                # Better fallback: try some basic grouping heuristics
                solution = self._create_smart_fallback(words)
            
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