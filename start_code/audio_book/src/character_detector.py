"""
Character Detection Module
Handles character and dialogue detection from text
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Dialogue:
    """Represents a dialogue in the text"""
    text: str
    speaker: Optional[str] = None
    position: int = 0
    context: str = ""


@dataclass
class Character:
    """Represents a character in the book"""
    name: str
    dialogues: List[str] = field(default_factory=list)
    dialogue_count: int = 0
    total_characters: int = 0


class CharacterDetector:
    """Detects characters and dialogues in text"""
    
    def __init__(self, config: dict):
        """
        Initialize character detector with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dialogue_patterns = [
            re.compile(pattern) for pattern in config['character_detection']['dialogue_patterns']
        ]
        self.min_dialogue_length = config['character_detection']['min_dialogue_length']
        self.max_dialogue_length = config['character_detection']['max_dialogue_length']
        
        self.logger = logging.getLogger(__name__)
    
    def extract_dialogues(self, text: str) -> List[Dialogue]:
        """
        Extract all dialogues with context from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of Dialogue objects
        """
        self.logger.info("Extracting dialogues from text")
        
        dialogues = []
        
        # Try each pattern
        for pattern in self.dialogue_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                dialogue_text = match.group(1).strip()
                
                # Check length constraints
                if len(dialogue_text) < self.min_dialogue_length:
                    continue
                if len(dialogue_text) > self.max_dialogue_length:
                    continue
                
                # Extract context (text before the dialogue)
                start_pos = match.start()
                context_start = max(0, start_pos - 200)
                context = text[context_start:start_pos].strip()
                
                # Try to identify speaker from context
                speaker = self._identify_speaker_from_context(dialogue_text, context, text)
                
                dialogue = Dialogue(
                    text=dialogue_text,
                    speaker=speaker,
                    position=start_pos,
                    context=context
                )
                
                dialogues.append(dialogue)
        
        self.logger.info(f"Extracted {len(dialogues)} dialogues")
        return dialogues
    
    def _identify_speaker_from_context(self, dialogue_text: str, 
                                     context: str, full_text: str) -> Optional[str]:
        """
        Try to identify speaker from surrounding text
        
        Args:
            dialogue_text: The dialogue text
            context: Text before the dialogue
            full_text: Full text for more context
            
        Returns:
            Speaker name or None
        """
        # Pattern 1: Look for name patterns before quotes
        # Chinese: "他说:" "李明道:" etc.
        chinese_speaker_pattern = re.compile(r'([一-龯A-Za-z]+)[:：]')
        match = chinese_speaker_pattern.search(context[-50:])
        if match:
            speaker_name = match.group(1)
            return speaker_name
        
        # English: "He said:", "Mary asked:", etc.
        english_speaker_pattern = re.compile(r'([A-Z][a-z]+)\s+(said|asked|replied|answered|exclaimed|whispered|murmured)')
        match = english_speaker_pattern.search(context[-100:])
        if match:
            speaker_name = match.group(1)
            return speaker_name
        
        # Pattern 2: Look for common pronouns that indicate speaker
        if re.search(r'\b(他|她|它|他|她|他|她)\b', context):
            # Chinese pronouns
            # This is a placeholder - actual speaker tracking would require context
            pass
        
        if re.search(r'\b(he|she|it)\b', context, re.IGNORECASE):
            # English pronouns
            pass
        
        # Pattern 3: Check if context ends with a name pattern
        # (Simplified for MVP)
        
        return None
    
    def build_character_database(self, dialogues: List[Dialogue]) -> Dict[str, Character]:
        """
        Build database of characters and their dialogues
        
        Args:
            dialogues: List of extracted dialogues
            
        Returns:
            Dict mapping name -> Character object
        """
        self.logger.info("Building character database")
        
        characters = {}
        unknown_speaker = "Unknown"
        
        for dialogue in dialogues:
            speaker_name = dialogue.speaker if dialogue.speaker else unknown_speaker
            
            if speaker_name not in characters:
                characters[speaker_name] = Character(
                    name=speaker_name,
                    dialogues=[],
                    dialogue_count=0,
                    total_characters=0
                )
            
            character = characters[speaker_name]
            character.dialogues.append(dialogue.text)
            character.dialogue_count += 1
            character.total_characters += len(dialogue.text)
        
        # Log character statistics
        for name, char in sorted(characters.items(), key=lambda x: x[1].dialogue_count, reverse=True):
            self.logger.info(
                f"Character '{name}': {char.dialogue_count} dialogues, "
                f"{char.total_characters} total characters"
            )
        
        self.logger.info(f"Found {len(characters)} unique characters")
        return characters
    
    def extract_text_segments(self, text: str, language: str) -> List[dict]:
        """
        Split text into segments (dialogue vs narration)
        Useful for TTS processing
        
        Args:
            text: Text to segment
            language: Language code ("zh" or "en")
            
        Returns:
            List of segment dicts with keys:
                - text: str
                - type: "dialogue" or "narration"
                - speaker: str or None
        """
        self.logger.debug("Extracting text segments")
        
        segments = []
        current_pos = 0
        
        # Try each pattern
        for pattern in self.dialogue_patterns:
            matches = list(pattern.finditer(text))
            
            for match in matches:
                # Add narration segment before this dialogue
                if match.start() > current_pos:
                    narration_text = text[current_pos:match.start()].strip()
                    if narration_text:
                        segments.append({
                            'text': narration_text,
                            'type': 'narration',
                            'speaker': None
                        })
                
                # Add dialogue segment
                dialogue_text = match.group(1).strip()
                
                # Try to identify speaker
                context_start = max(0, match.start() - 200)
                context = text[context_start:match.start()].strip()
                speaker = self._identify_speaker_from_context(dialogue_text, context, text)
                
                segments.append({
                    'text': dialogue_text,
                    'type': 'dialogue',
                    'speaker': speaker
                })
                
                current_pos = match.end()
        
        # Add remaining narration
        if current_pos < len(text):
            narration_text = text[current_pos:].strip()
            if narration_text:
                segments.append({
                    'text': narration_text,
                    'type': 'narration',
                    'speaker': None
                })
        
        self.logger.debug(f"Extracted {len(segments)} segments")
        return segments
