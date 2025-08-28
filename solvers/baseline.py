from typing import List
from .base import BaseSolver


class BaselineSolver(BaseSolver):
    """Baseline solver with a simple prompt."""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the baseline solver."""
        return """You are an expert at solving NYT Connections puzzles. 
        Given 16 words, you need to group them into 4 groups of 4 words each.
        
        CRITICAL RULES:
        - Each word must appear in EXACTLY ONE group (no duplicates across groups)
        - Each group must contain EXACTLY 4 words
        - All 16 words must be used (no words left out, no words repeated)
        
        Consider various types of connections including:
        - Literal meanings and categories
        - Wordplay (palindromes, homophones, rhymes)
        - Words that can precede or follow a common word
        - Pop culture references
        - Common phrases or idioms
        
        Be concise but clear in your reasoning."""
    
    def get_user_message(self, words: List[str]) -> str:
        """Format the user message for the baseline solver."""
        return f"""Group these 16 words into 4 groups of 4 words each. 
    Each group should share a common theme or connection.
    
    Words: {', '.join(words)}
    
    Provide a clear, concise reason for each grouping."""