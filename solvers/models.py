from typing import List
from pydantic import BaseModel, Field, field_validator


class GroupSolution(BaseModel):
    """A single group of 4 words with their connecting reason."""
    words: List[str] = Field(description="Exactly 4 words that belong together", min_length=4, max_length=4)
    reason: str = Field(description="The reason/theme connecting these 4 words")
    
    @field_validator('words')
    @classmethod
    def validate_words_count(cls, v):
        if len(v) != 4:
            raise ValueError(f"Each group must have exactly 4 words, got {len(v)}")
        return v


class PuzzleSolution(BaseModel):
    """Complete solution to a Connections puzzle."""
    groups: List[GroupSolution] = Field(
        description="Exactly 4 groups of 4 words each",
        min_length=4,
        max_length=4
    )
    
    @field_validator('groups')
    @classmethod
    def validate_groups(cls, v):
        if len(v) != 4:
            raise ValueError(f"Must have exactly 4 groups, got {len(v)}")
        
        # Check that all words are unique across groups
        all_words = []
        for group in v:
            all_words.extend(group.words)
        
        if len(all_words) != len(set(all_words)):
            raise ValueError("Words must be unique across all groups")
        
        if len(all_words) != 16:
            raise ValueError(f"Total must be 16 unique words, got {len(set(all_words))}")
        
        return v