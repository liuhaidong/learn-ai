"""
TTS Engine Module
Handles text-to-speech generation using Coqui TTS
"""

import os
import logging
from pathlib import Path
from typing import List, Dict
from TTS.api import TTS
from pydub import AudioSegment
from .pdf_processor import Chapter
from .character_detector import CharacterDetector
from .voice_mapper import VoiceMapper
from .llm_analyzer import CharacterProfile


class TTSEngine:
    """Generates speech using Coqui TTS"""
    
    def __init__(self, config: dict):
        """
        Initialize TTS engine with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config['tts']['model']
        self.device = config['tts']['device']
        self.auto_download = config['tts']['auto_download']
        self.output_format = config['output']['format']
        self.sample_rate = config['output']['sample_rate']
        self.chapter_padding = config['output']['chapter_padding']
        self.temp_dir = Path(config['output']['temp_dir'])
        
        self.logger = logging.getLogger(__name__)
        
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Coqui TTS
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize Coqui TTS model"""
        try:
            self.logger.info(f"Initializing TTS model: {self.model_name}")
            
            # Initialize TTS
            self.tts = TTS(self.model_name).to(self.device)
            
            # List available speakers
            speakers = self.tts.speakers if hasattr(self.tts, 'speakers') else []
            self.logger.info(f"TTS initialized with {len(speakers)} speakers")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    def text_to_speech(self, text: str, language: str, 
                     voice: str, output_path: str,
                     speed: float = 1.0) -> str:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            language: Language code ("zh" or "en")
            voice: Voice ID (e.g., "zh-CN-XiaoxiaoNeural")
            output_path: Path to save audio file
            speed: Speed adjustment (1.0 = normal)
            
        Returns:
            Path to generated audio file
        """
        self.logger.debug(f"Generating TTS for text ({len(text)} chars)")
        
        try:
            # Map language code for Coqui
            lang_code = "zh-cn" if language == "zh" else "en"
            
            # Generate speech
            self.tts.tts_to_file(
                text=text,
                speaker=voice,
                language=lang_code,
                file_path=output_path,
                speed=speed
            )
            
            self.logger.debug(f"TTS generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            raise
    
    def generate_chapter_audio(self, chapter: Chapter, 
                           character_profiles: Dict[str, CharacterProfile],
                           character_detector: CharacterDetector,
                           voice_mapper: VoiceMapper,
                           output_dir: Path) -> str:
        """
        Generate audio for entire chapter
        
        Args:
            chapter: Chapter object with content
            character_profiles: Dict of character profiles
            character_detector: CharacterDetector instance
            voice_mapper: VoiceMapper instance
            output_dir: Directory to save chapter audio
            
        Returns:
            Path to generated chapter audio file
        """
        self.logger.info(f"Generating audio for chapter {chapter.chapter_number}: {chapter.title}")
        
        # Detect language
        language = self._detect_language(chapter.content)
        self.logger.debug(f"Chapter language: {language}")
        
        # Split text into segments
        segments = character_detector.extract_text_segments(chapter.content, language)
        self.logger.info(f"Chapter has {len(segments)} segments")
        
        # Generate audio for each segment
        segment_files = []
        
        for i, segment in enumerate(segments):
            self.logger.debug(f"Processing segment {i + 1}/{len(segments)}")
            
            try:
                segment_file = self._generate_segment_audio(
                    segment=segment,
                    chapter_number=chapter.chapter_number,
                    segment_number=i,
                    language=language,
                    character_profiles=character_profiles,
                    voice_mapper=voice_mapper
                )
                segment_files.append(segment_file)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate segment {i}: {e}, skipping")
                continue
        
        if not segment_files:
            self.logger.error("No segments generated successfully")
            raise RuntimeError("Failed to generate any audio segments")
        
        # Concatenate all segments
        chapter_file = self._concatenate_segments(
            segment_files=segment_files,
            chapter=chapter,
            output_dir=output_dir
        )
        
        # Cleanup temp files
        self._cleanup_temp_files(segment_files)
        
        self.logger.info(f"Chapter audio generated: {chapter_file}")
        return chapter_file
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection for TTS
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code ("zh" or "en")
        """
        # Simple heuristic: count Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        
        if chinese_chars / len(text) > 0.3:
            return "zh"
        else:
            return "en"
    
    def _generate_segment_audio(self, segment: dict, chapter_number: int,
                           segment_number: int, language: str,
                           character_profiles: Dict[str, CharacterProfile],
                           voice_mapper: VoiceMapper) -> str:
        """
        Generate audio for a single segment
        
        Args:
            segment: Segment dict with text, type, speaker
            chapter_number: Chapter number
            segment_number: Segment index
            language: Language code
            character_profiles: Dict of character profiles
            voice_mapper: VoiceMapper instance
            
        Returns:
            Path to segment audio file
        """
        text = segment['text']
        segment_type = segment['type']
        speaker_name = segment.get('speaker')
        
        # Determine voice
        if segment_type == 'dialogue' and speaker_name:
            # Use character-specific voice
            if speaker_name in character_profiles:
                profile = character_profiles[speaker_name]
                voice = voice_mapper.get_voice_for_character(speaker_name, profile, language)
                params = voice_mapper.get_voice_parameters(profile.age_group)
            else:
                # Unknown character, use narrator voice
                voice = voice_mapper.get_default_narrator_voice(language)
                params = voice_mapper.get_voice_parameters('adult')
        else:
            # Narration
            voice = voice_mapper.get_default_narrator_voice(language)
            params = voice_mapper.get_voice_parameters('adult')
        
        # Generate audio
        output_path = self.temp_dir / f"ch{chapter_number}_seg{segment_number}.mp3"
        
        self.text_to_speech(
            text=text,
            language=language,
            voice=voice,
            output_path=str(output_path),
            speed=params.get('rate', 1.0)
        )
        
        return str(output_path)
    
    def _concatenate_segments(self, segment_files: List[str],
                          chapter: Chapter, output_dir: Path) -> str:
        """
        Concatenate audio segments into chapter file
        
        Args:
            segment_files: List of segment audio file paths
            chapter: Chapter object
            output_dir: Output directory
            
        Returns:
            Path to chapter audio file
        """
        self.logger.debug(f"Concatenating {len(segment_files)} segments")
        
        # Load all segments
        segments_audio = []
        for segment_file in segment_files:
            try:
                audio = AudioSegment.from_mp3(segment_file)
                segments_audio.append(audio)
            except Exception as e:
                self.logger.warning(f"Failed to load segment {segment_file}: {e}")
                continue
        
        if not segments_audio:
            raise RuntimeError("No valid segments to concatenate")
        
        # Create silence padding
        silence = AudioSegment.silent(duration=int(self.chapter_padding * 1000))
        
        # Concatenate with padding
        chapter_audio = AudioSegment.empty()
        for i, segment in enumerate(segments_audio):
            chapter_audio += segment
            if i < len(segments_audio) - 1:
                chapter_audio += silence
        
        # Save chapter audio
        output_file = output_dir / f"chapter_{chapter.chapter_number:02d}.{self.output_format}"
        chapter_audio.export(
            str(output_file),
            format=self.output_format,
            bitrate=self.config['output']['bitrate']
        )
        
        return str(output_file)
    
    def _cleanup_temp_files(self, temp_files: List[str]):
        """
        Clean up temporary files
        
        Args:
            temp_files: List of temp file paths
        """
        if not self.config['output']['cleanup_temp']:
            return
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
