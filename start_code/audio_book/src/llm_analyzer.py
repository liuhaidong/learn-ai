"""
LLM Analyzer Module
Uses DeepSeek API to analyze characters (gender, age, etc.)
"""

import json
import time
import logging
from typing import List, Dict
from dataclasses import dataclass
from openai import OpenAI
from .character_detector import Character


@dataclass
class CharacterProfile:
    """Profile of a character analyzed by LLM"""
    name: str
    gender: str  # "male", "female", "unknown"
    age_group: str  # "child", "young_adult", "adult", "elderly"
    confidence: float
    reasoning: str


class LLMAnalyzer:
    """Analyzes characters using DeepSeek API"""
    
    def __init__(self, config: dict):
        """
        Initialize LLM analyzer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.provider = config['llm']['provider']
        self.api_key = config['llm']['api_key'].replace('${DEEPSEEK_API_KEY}', '').strip()
        self.base_url = config['llm']['base_url']
        self.model = config['llm']['model']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']
        self.retry_attempts = config['llm']['retry_attempts']
        self.retry_delay = config['llm']['retry_delay']
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client (compatible with DeepSeek)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.logger.info(f"Initialized LLM analyzer with {self.provider}")
    
    def analyze_character(self, character: Character) -> CharacterProfile:
        """
        Analyze a single character's dialogues to determine gender and age
        
        Args:
            character: Character object with dialogues
            
        Returns:
            CharacterProfile object
        """
        self.logger.info(f"Analyzing character: {character.name}")
        
        # Prepare dialogues for analysis (use first 10 dialogues as sample)
        sample_dialogues = character.dialogues[:10]
        dialogues_text = '\n'.join([f'- "{d}"' for d in sample_dialogues])
        
        # Create prompt
        prompt = self._create_analysis_prompt(character.name, dialogues_text)
        
        # Retry logic
        for attempt in range(self.retry_attempts):
            try:
                response = self._call_llm_api(prompt)
                
                # Parse JSON response
                profile_data = self._parse_response(response)
                
                profile = CharacterProfile(
                    name=character.name,
                    gender=profile_data.get('gender', 'unknown'),
                    age_group=profile_data.get('age_group', 'adult'),
                    confidence=profile_data.get('confidence', 0.5),
                    reasoning=profile_data.get('reasoning', '')
                )
                
                self.logger.info(
                    f"Character '{character.name}': gender={profile.gender}, "
                    f"age_group={profile.age_group}, confidence={profile.confidence:.2f}"
                )
                
                return profile
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for character '{character.name}': {e}"
                )
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"All attempts failed for character '{character.name}'")
                    # Return default profile
                    return CharacterProfile(
                        name=character.name,
                        gender="unknown",
                        age_group="adult",
                        confidence=0.0,
                        reasoning=f"Analysis failed: {str(e)}"
                    )
        
        # Should never reach here, but for type safety
        return CharacterProfile(
            name=character.name,
            gender="unknown",
            age_group="adult",
            confidence=0.0,
            reasoning="Unexpected error in analysis"
        )
    
    def analyze_characters_batch(self, characters: Dict[str, Character]) -> Dict[str, CharacterProfile]:
        """
        Analyze multiple characters sequentially
        
        Args:
            characters: Dict mapping name -> Character object
            
        Returns:
            Dict mapping name -> CharacterProfile object
        """
        self.logger.info(f"Analyzing {len(characters)} characters")
        
        profiles = {}
        
        for name, character in characters.items():
            profile = self.analyze_character(character)
            profiles[name] = profile
            
            # Small delay between requests to avoid rate limiting
            time.sleep(1)
        
        self.logger.info("Finished character analysis")
        return profiles
    
    def _create_analysis_prompt(self, character_name: str, dialogues_text: str) -> str:
        """
        Create prompt for character analysis
        
        Args:
            character_name: Name of character
            dialogues_text: Sample dialogues
            
        Returns:
            Prompt string
        """
        prompt = f"""You are a literary analyst. Based on the following dialogues from a novel, determine the character's gender and approximate age group.

Character name: {character_name}

Dialogues:
{dialogues_text}

Determine:
1. Gender: male, female, or unknown
2. Age group: child, young_adult, adult, or elderly

Respond in JSON format only (no other text):
{{
    "gender": "male|female|unknown",
    "age_group": "child|young_adult|adult|elderly",
    "confidence": 0.8,
    "reasoning": "Brief explanation of your reasoning..."
}}

Confidence should be between 0.0 and 1.0 indicating how certain you are."""
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call LLM API
        
        Args:
            prompt: Prompt string
            
        Returns:
            Response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful literary analyst. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM API returned None content")
            return content
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_response(self, response: str) -> dict:
        """
        Parse JSON response from LLM
        
        Args:
            response: Response string from LLM
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try to extract JSON from response (in case there's extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            # Validate required fields
            if 'gender' not in data:
                data['gender'] = 'unknown'
            if 'age_group' not in data:
                data['age_group'] = 'adult'
            if 'confidence' not in data:
                data['confidence'] = 0.5
            if 'reasoning' not in data:
                data['reasoning'] = ''
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Response was: {response}")
            # Return default values
            return {
                'gender': 'unknown',
                'age_group': 'adult',
                'confidence': 0.0,
                'reasoning': 'Failed to parse response'
            }
