import os
import json
from typing import List, Dict, Any, Optional
import dspy
from .base import BaseSolver
from .models import PuzzleSolution, GroupSolution


class ThemeAnalysisSignature(dspy.Signature):
    """Analyze words for potential thematic connections and patterns."""
    
    words = dspy.InputField(desc="16 words to analyze for patterns")
    literal_themes = dspy.OutputField(desc="Literal categories found (e.g., colors, animals, tools)")
    wordplay_patterns = dspy.OutputField(desc="Wordplay patterns (homophones, rhymes, anagrams)")
    phrase_patterns = dspy.OutputField(desc="Common phrases or idioms that could be completed")
    cultural_references = dspy.OutputField(desc="Pop culture, geography, or specialized knowledge themes")
    analysis_summary = dspy.OutputField(desc="Overall theme analysis with confidence levels")


class HypothesisGenerationSignature(dspy.Signature):
    """Generate candidate groupings based on theme analysis."""
    
    words = dspy.InputField(desc="16 words to group")
    theme_analysis = dspy.InputField(desc="Analysis of themes and patterns")
    hypothesis1 = dspy.OutputField(desc="First candidate grouping: 4 words (comma-separated)")
    hypothesis1_reason = dspy.OutputField(desc="Reasoning for hypothesis 1")
    hypothesis2 = dspy.OutputField(desc="Second candidate grouping: 4 words (comma-separated)")
    hypothesis2_reason = dspy.OutputField(desc="Reasoning for hypothesis 2")
    hypothesis3 = dspy.OutputField(desc="Third candidate grouping: 4 words (comma-separated)")
    hypothesis3_reason = dspy.OutputField(desc="Reasoning for hypothesis 3")
    hypothesis4 = dspy.OutputField(desc="Fourth candidate grouping: 4 words (comma-separated)")
    hypothesis4_reason = dspy.OutputField(desc="Reasoning for hypothesis 4")
    hypothesis5 = dspy.OutputField(desc="Fifth candidate grouping: 4 words (comma-separated)")
    hypothesis5_reason = dspy.OutputField(desc="Reasoning for hypothesis 5")


class ValidationSignature(dspy.Signature):
    """Validate candidate groups for conflicts and coverage."""
    
    all_words = dspy.InputField(desc="All 16 words in the puzzle")
    candidate_groups = dspy.InputField(desc="Candidate groups to validate")
    conflicts = dspy.OutputField(desc="Any word overlaps or conflicts between groups")
    missing_words = dspy.OutputField(desc="Words not covered by any group")
    group_quality = dspy.OutputField(desc="Quality assessment of each group (strong/medium/weak)")
    validation_result = dspy.OutputField(desc="Overall validation: pass/fail with reasons")


class RefinementSignature(dspy.Signature):
    """Refine groups based on validation to produce final solution."""
    
    words = dspy.InputField(desc="All 16 words in the puzzle")
    candidate_groups = dspy.InputField(desc="Candidate groups that were validated")
    validation_result = dspy.InputField(desc="Validation results and issues found")
    final_group1 = dspy.OutputField(desc="Final group 1: exactly 4 words (comma-separated)")
    final_group1_reason = dspy.OutputField(desc="Connection theme for final group 1")
    final_group2 = dspy.OutputField(desc="Final group 2: exactly 4 words (comma-separated)")
    final_group2_reason = dspy.OutputField(desc="Connection theme for final group 2")
    final_group3 = dspy.OutputField(desc="Final group 3: exactly 4 words (comma-separated)")
    final_group3_reason = dspy.OutputField(desc="Connection theme for final group 3")
    final_group4 = dspy.OutputField(desc="Final group 4: exactly 4 words (comma-separated)")
    final_group4_reason = dspy.OutputField(desc="Connection theme for final group 4")


class MultiStageConnectionsSolver(dspy.Module):
    """Multi-stage reasoning pipeline for solving Connections puzzles."""
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(ThemeAnalysisSignature)
        self.generate_hypotheses = dspy.ChainOfThought(HypothesisGenerationSignature)
        self.validate = dspy.ChainOfThought(ValidationSignature)
        self.refine = dspy.ChainOfThought(RefinementSignature)
    
    def forward(self, words: str):
        """Execute the multi-stage reasoning pipeline."""
        
        print("\n🔍 Stage 1: Theme Analysis")
        theme_analysis = self.analyze(words=words)
        
        print("\n💡 Stage 2: Hypothesis Generation")
        hypotheses = self.generate_hypotheses(
            words=words,
            theme_analysis=f"""
            Literal themes: {theme_analysis.literal_themes}
            Wordplay: {theme_analysis.wordplay_patterns}
            Phrases: {theme_analysis.phrase_patterns}
            Cultural: {theme_analysis.cultural_references}
            Summary: {theme_analysis.analysis_summary}
            """
        )
        
        # Format candidate groups for validation
        candidate_groups = []
        for i in range(1, 6):  # We generate 5 hypotheses
            group = getattr(hypotheses, f"hypothesis{i}", "")
            reason = getattr(hypotheses, f"hypothesis{i}_reason", "")
            if group:
                candidate_groups.append(f"Group {i}: {group} (Reason: {reason})")
        
        print("\n✅ Stage 3: Validation")
        validation = self.validate(
            all_words=words,
            candidate_groups="\n".join(candidate_groups)
        )
        
        print("\n🎯 Stage 4: Refinement")
        final_solution = self.refine(
            words=words,
            candidate_groups="\n".join(candidate_groups),
            validation_result=f"""
            Conflicts: {validation.conflicts}
            Missing words: {validation.missing_words}
            Quality: {validation.group_quality}
            Result: {validation.validation_result}
            """
        )
        
        return final_solution


class MultiStageSolver(BaseSolver):
    """Multi-stage DSPy solver with explicit reasoning pipeline.
    
    This solver breaks down the solving process into distinct stages:
    1. Theme Analysis - Identify potential patterns and connections
    2. Hypothesis Generation - Create multiple candidate groupings
    3. Validation - Check for conflicts and completeness
    4. Refinement - Produce final optimized solution
    """
    
    def __init__(self, examples: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.examples = examples or []
        self.solver = None
        self.lm = None
    
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return "You are an expert at solving NYT Connections puzzles using systematic multi-stage reasoning."
    
    def get_user_message(self, words: List[str]) -> str:
        """Format user message."""
        return ", ".join(words)
    
    def format_example_for_stages(self, example: Dict) -> Dict[str, dspy.Example]:
        """Format an example for each stage of the pipeline."""
        words_str = ", ".join(example['words'])
        groups = example['solution']['groups']
        
        # Theme Analysis Example
        theme_example = dspy.Example(
            words=words_str,
            literal_themes="Animals, colors, tools, emotions",
            wordplay_patterns="Rhyming pairs, homophones detected",
            phrase_patterns="'Break a ___', 'Take a ___' patterns",
            cultural_references="Movie titles, brand names",
            analysis_summary="Strong literal categories with some wordplay elements"
        ).with_inputs('words')
        
        # Hypothesis Generation Example
        hypothesis_example = dspy.Example(
            words=words_str,
            theme_analysis="Multiple strong themes identified",
            hypothesis1=", ".join(groups[0]['words']),
            hypothesis1_reason=groups[0]['reason'],
            hypothesis2=", ".join(groups[1]['words']),
            hypothesis2_reason=groups[1]['reason'],
            hypothesis3=", ".join(groups[2]['words']),
            hypothesis3_reason=groups[2]['reason'],
            hypothesis4=", ".join(groups[3]['words']),
            hypothesis4_reason=groups[3]['reason'],
            hypothesis5=", ".join(groups[0]['words'][:2] + groups[1]['words'][:2]),
            hypothesis5_reason="Alternative grouping"
        ).with_inputs('words', 'theme_analysis')
        
        # Validation Example
        validation_example = dspy.Example(
            all_words=words_str,
            candidate_groups="Group candidates listed",
            conflicts="No overlapping words detected",
            missing_words="None",
            group_quality="Group 1: strong, Group 2: strong, Group 3: medium, Group 4: strong",
            validation_result="Pass - all words covered, no conflicts"
        ).with_inputs('all_words', 'candidate_groups')
        
        # Refinement Example
        refinement_example = dspy.Example(
            words=words_str,
            candidate_groups="Validated groups",
            validation_result="Pass with minor adjustments needed",
            final_group1=", ".join(groups[0]['words']),
            final_group1_reason=groups[0]['reason'],
            final_group2=", ".join(groups[1]['words']),
            final_group2_reason=groups[1]['reason'],
            final_group3=", ".join(groups[2]['words']),
            final_group3_reason=groups[2]['reason'],
            final_group4=", ".join(groups[3]['words']),
            final_group4_reason=groups[3]['reason']
        ).with_inputs('words', 'candidate_groups', 'validation_result')
        
        return {
            'theme': theme_example,
            'hypothesis': hypothesis_example,
            'validation': validation_example,
            'refinement': refinement_example
        }
    
    def solve(self, words: List[str], use_api: bool = False, model: str = "gpt-4o-mini") -> PuzzleSolution:
        """
        Solve using multi-stage reasoning pipeline.
        
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
                    max_tokens=1000,
                    temperature=0.7
                )
                dspy.settings.configure(lm=self.lm)
            
            # Initialize solver if needed
            if not self.solver:
                self.solver = MultiStageConnectionsSolver()
            
            # Add examples if available
            if self.examples and len(self.examples) > 0:
                example = self.examples[0]  # Use first example for demonstration
                stage_examples = self.format_example_for_stages(example)
                
                # Set examples for each stage
                self.solver.analyze.demos = [stage_examples['theme']]
                self.solver.generate_hypotheses.demos = [stage_examples['hypothesis']]
                self.solver.validate.demos = [stage_examples['validation']]
                self.solver.refine.demos = [stage_examples['refinement']]
            
            # Solve the puzzle
            words_str = ", ".join(words)
            
            try:
                # Execute first 3 stages of pipeline
                print("\n🔍 Stage 1: Theme Analysis")
                theme_analysis = self.solver.analyze(words=words_str)
                
                print("\n💡 Stage 2: Hypothesis Generation")
                hypotheses = self.solver.generate_hypotheses(
                    words=words_str,
                    theme_analysis=f"""
                    Literal themes: {theme_analysis.literal_themes}
                    Wordplay: {theme_analysis.wordplay_patterns}
                    Phrases: {theme_analysis.phrase_patterns}
                    Cultural: {theme_analysis.cultural_references}
                    Summary: {theme_analysis.analysis_summary}
                    """
                )
                
                # Format candidate groups for validation
                candidate_groups = []
                for i in range(1, 6):  # We generate 5 hypotheses
                    group = getattr(hypotheses, f"hypothesis{i}", "")
                    reason = getattr(hypotheses, f"hypothesis{i}_reason", "")
                    if group:
                        candidate_groups.append(f"Group {i}: {group} (Reason: {reason})")
                
                print("\n✅ Stage 3: Validation")
                validation = self.solver.validate(
                    all_words=words_str,
                    candidate_groups="\n".join(candidate_groups)
                )
                
                # Refinement stage with retry logic
                print("\n🎯 Stage 4: Refinement")
                max_refinement_retries = 3
                best_solution = None
                
                for attempt in range(max_refinement_retries):
                    # Add stronger guidance on retry
                    validation_feedback = f"""
                    Conflicts: {validation.conflicts}
                    Missing words: {validation.missing_words}
                    Quality: {validation.group_quality}
                    Result: {validation.validation_result}
                    """
                    
                    if attempt > 0:
                        validation_feedback += f"\n\nATTEMPT {attempt + 1}: Previous refinement failed validation. "
                        validation_feedback += "CRITICAL: Each group must have EXACTLY 4 words, all 16 words must be used exactly once."
                    
                    final_solution = self.solver.refine(
                        words=words_str,
                        candidate_groups="\n".join(candidate_groups),
                        validation_result=validation_feedback
                    )
                    
                    # Validate the refined solution
                    validation_issues = []
                    all_words_used = []
                    
                    for i in range(1, 5):
                        group_words = getattr(final_solution, f"final_group{i}", "")
                        words_list = [w.strip() for w in group_words.split(",") if w.strip()]
                        
                        if len(words_list) != 4:
                            validation_issues.append(f"Group {i} has {len(words_list)} words")
                        
                        all_words_used.extend(words_list)
                    
                    # Check for issues
                    if len(all_words_used) != len(set(all_words_used)):
                        duplicates = [w for w in set(all_words_used) if all_words_used.count(w) > 1]
                        validation_issues.append(f"Duplicates: {duplicates}")
                    
                    if len(set(all_words_used)) != 16:
                        validation_issues.append(f"Only {len(set(all_words_used))}/16 words used")
                    
                    if not validation_issues:
                        best_solution = final_solution
                        break
                    elif attempt < max_refinement_retries - 1:
                        print(f"  Refinement attempt {attempt + 1} had issues: {validation_issues}")
                        print("  Retrying refinement stage...")
                    else:
                        print(f"  Warning: Using solution with issues after {max_refinement_retries} refinement attempts")
                        best_solution = final_solution
                
                pred = best_solution
                
                # Track API costs from DSPy multi-stage pipeline
                if hasattr(self.lm, '_total_cost') and self.lm._total_cost > 0:
                    # DSPy tracks costs internally, log it
                    self.log_api_cost(model, 0, 0, self.lm._total_cost)
                    print(f"Multi-stage DSPy API cost: ${self.lm._total_cost:.4f}")
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
                            
                            pricing = self.MODEL_PRICING.get(model, self.MODEL_PRICING["gpt-4o-mini"])
                            cost = (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]
                            total_cost += cost
                    
                    if total_cost > 0:
                        self.log_api_cost(model, total_prompt_tokens, total_completion_tokens, total_cost)
                        print(f"Multi-stage DSPy API cost: ${total_cost:.4f}")
                
                # Parse the final refined output
                groups = []
                for i in range(1, 5):
                    group_words = getattr(pred, f"final_group{i}", "")
                    group_reason = getattr(pred, f"final_group{i}_reason", "")
                    
                    # Clean and parse words
                    words_list = [w.strip() for w in group_words.split(",")]
                    
                    # Ensure exactly 4 words
                    if len(words_list) > 4:
                        words_list = words_list[:4]
                    elif len(words_list) < 4:
                        # Fill with remaining unused words
                        used_words = set()
                        for j in range(1, i):
                            prev_words = getattr(pred, f"final_group{j}", "")
                            used_words.update(w.strip() for w in prev_words.split(","))
                        
                        remaining = [w for w in words if w not in used_words]
                        words_list.extend(remaining[:4-len(words_list)])
                    
                    groups.append(GroupSolution(
                        words=words_list,
                        reason=group_reason or f"Group {i}"
                    ))
                
                solution = PuzzleSolution(groups=groups)
                
                print("\n" + "="*50)
                print("Multi-Stage Pipeline Complete!")
                print("="*50)
                
            except Exception as e:
                print(f"Multi-stage solving failed: {e}")
                # Fallback to simple solution
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
            # Test mode
            print("="*80)
            print("Multi-Stage Solver - Test Mode")
            print("Pipeline: Theme Analysis → Hypothesis Generation → Validation → Refinement")
            print("="*80)
            
            return PuzzleSolution(
                groups=[
                    GroupSolution(words=list(words[:4]), reason="dummy group 1"),
                    GroupSolution(words=list(words[4:8]), reason="dummy group 2"),
                    GroupSolution(words=list(words[8:12]), reason="dummy group 3"),
                    GroupSolution(words=list(words[12:16]), reason="dummy group 4")
                ]
            )