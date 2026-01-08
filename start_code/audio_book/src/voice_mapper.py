"""
Voice Mapper Module
Maps characters to appropriate TTS voices
"""

import logging
from typing import Dict, Optional
from .llm_analyzer import CharacterProfile


class VoiceMapper:
    """Maps character profiles to TTS voices"""
    
    def __init__(self, config: dict):
        """
        Initialize voice mapper with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.default_voices = config['tts']['default_voice']
        self.voice_mapping = config['tts']['voice_mapping']
        self.voice_parameters = config['tts']['voice_parameters']
        
        self.logger = logging.getLogger(__name__)
    
    def get_voice_for_character(self, character_name: str,
                             profile: CharacterProfile,
                             language: str) -> str:
        """
        Get appropriate voice ID for character
        
        Args:
            character_name: Character name
            profile: CharacterProfile from LLM analysis
            language: Language code ("zh" or "en")
            
        Returns:
            Voice ID (e.g., "zh-CN-YunxiNeural")
        """
        self.logger.debug(f"Getting voice for character '{character_name}'")
        
        # Map gender + age_group to voice category
        voice_category = self._map_profile_to_category(profile)
        
        # Get voice from mapping
        if voice_category in self.voice_mapping:
            voice_id = self.voice_mapping[voice_category].get(language)
            
            if voice_id:
                self.logger.debug(
                    f"Mapped character '{character_name}' ({profile.gender}, "
                    f"{profile.age_group}) to voice '{voice_id}'"
                )
                return voice_id
        
        # Fallback to default narrator voice
        default_voice = self.get_default_narrator_voice(language)
        self.logger.warning(
            f"No voice mapping found for '{voice_category}', using default '{default_voice}'"
        )
        return default_voice
    
    def _map_profile_to_category(self, profile: CharacterProfile) -> str:
        """
        Map character profile to voice category
        
        Args:
            profile: CharacterProfile
            
        Returns:
            Voice category string
        """
        gender = profile.gender
        age_group = profile.age_group
        
        # Map combinations
        if gender == 'unknown':
            return 'unknown'
        
        # Build category name
        category = age_group
        
        # Adjust for gender-specific categories
        if gender == 'male':
            if age_group in ['child', 'young_adult', 'adult', 'elderly']:
                category = f"{age_group}_male"
        elif gender == 'female':
            if age_group in ['child', 'young_adult', 'adult', 'elderly']:
                category = f"{age_group}_female"
        
        # If specific category not found, try gender-neutral
        if category not in self.voice_mapping:
            return gender
        
        return category
    
    def get_default_narrator_voice(self, language: str) -> str:
        """
        Get default narrator voice
        
        Args:
            language: Language code ("zh" or "en")
            
        Returns:
            Voice ID for narration
        """
        return self.default_voices.get(language, 'en-US-JennyNeural')
    
    def get_voice_parameters(self, age_group: str) -> dict:
        """
        Get TTS parameters adjustments based on age group
        
        Args:
            age_group: Age group string
            
        Returns:
            dict with pitch, rate, volume adjustments
        """
        if age_group in self.voice_parameters:
            return self.voice_parameters[age_group]
        
        # Default parameters
        return {
            'pitch': 1.0,
            'rate': 1.0,
            'volume': 1.0
        }
