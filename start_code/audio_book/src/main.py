"""
Main entry point for PDF-to-Audiobook converter
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
from typing import Optional

from .pdf_processor import PDFProcessor
from .character_detector import CharacterDetector
from .llm_analyzer import LLMAnalyzer
from .voice_mapper import VoiceMapper
from .tts_engine import TTSEngine


def setup_logging(config: dict):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
    """
    log_level = config['logging']['level']
    log_file = config['logging'].get('file')
    log_console = config['logging'].get('console', True)
    
    # Configure logging format
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    
    # Remove console handler if not needed
    if not log_console:
        logging.getLogger().handlers = [h for h in logging.getLogger().handlers 
                                      if not isinstance(h, logging.StreamHandler)]


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables in config
    config = _replace_env_vars(config)
    
    return config


def _replace_env_vars(config: dict) -> dict:
    """
    Replace environment variables in configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with replaced values
    """
    import os
    
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR_NAME} with environment variable
        import re
        match = re.match(r'\$\{(\w+)\}', config)
        if match:
            var_name = match.group(1)
            return os.getenv(var_name, config)
        return config
    else:
        return config


def process_pdf(pdf_path: str, output_dir: str, config: dict,
                language: Optional[str] = None,
                chapters: Optional[str] = None):
    """
    Process PDF file and generate audiobook
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        config: Configuration dictionary
        language: Language override ("zh" or "en" or "auto")
        chapters: Chapter range to process (e.g., "1-5", "all")
    """
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("PDF-to-Audiobook Converter MVP")
    logger.info("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    try:
        pdf_processor = PDFProcessor(config)
        character_detector = CharacterDetector(config)
        
        logger.info("Initializing LLM analyzer...")
        llm_analyzer = LLMAnalyzer(config)
        
        logger.info("Initializing TTS engine...")
        tts_engine = TTSEngine(config)
        voice_mapper = VoiceMapper(config)
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    # Step 1: Extract text from PDF
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Extracting text from PDF")
    logger.info("="*80)
    
    try:
        text = pdf_processor.extract_text(pdf_path)
        logger.info(f"Extracted {len(text)} characters")
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        logger.error("Cannot continue without PDF text")
        return
    
    # Step 2: Detect language
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Detecting language")
    logger.info("="*80)
    
    if language == "auto" or not language:
        language = pdf_processor.detect_language(text)
    
    logger.info(f"Language detected: {language}")
    
    # Step 3: Split into chapters
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Splitting text into chapters")
    logger.info("="*80)
    
    try:
        chapters_list = pdf_processor.split_by_chapters(text)
        logger.info(f"Created {len(chapters_list)} chapters")
    except Exception as e:
        logger.error(f"Failed to split chapters: {e}")
        logger.error("Cannot continue without chapters")
        return
    
    # Filter chapters if range specified
    if chapters and chapters != "all":
        try:
            if '-' in chapters:
                start, end = map(int, chapters.split('-'))
                chapters_list = chapters_list[start-1:end]
            else:
                chapter_num = int(chapters)
                chapters_list = [chapters_list[chapter_num-1]]
            logger.info(f"Processing {len(chapters_list)} chapters: {chapters}")
        except Exception as e:
            logger.error(f"Invalid chapter range '{chapters}': {e}, processing all chapters")
    
    # Step 4: Detect characters and dialogues
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Detecting characters and dialogues")
    logger.info("="*80)
    
    try:
        all_dialogues = []
        for chapter in chapters_list:
            dialogues = character_detector.extract_dialogues(chapter.content)
            all_dialogues.extend(dialogues)
        
        logger.info(f"Extracted {len(all_dialogues)} total dialogues")
    except Exception as e:
        logger.error(f"Failed to extract dialogues: {e}")
        logger.warning("Continuing without character detection...")
        all_dialogues = []
    
    # Step 5: Build character database
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Building character database")
    logger.info("="*80)
    
    try:
        characters = character_detector.build_character_database(all_dialogues)
        logger.info(f"Found {len(characters)} unique characters")
    except Exception as e:
        logger.error(f"Failed to build character database: {e}")
        logger.warning("Continuing without character analysis...")
        characters = {}
    
    # Step 6: Analyze characters with LLM
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Analyzing characters with LLM")
    logger.info("="*80)
    
    character_profiles = {}
    
    if characters:
        try:
            character_profiles = llm_analyzer.analyze_characters_batch(characters)
            logger.info(f"Analyzed {len(character_profiles)} characters")
        except Exception as e:
            logger.error(f"Failed to analyze characters: {e}")
            logger.warning("Continuing without character profiles...")
    
    # Save character profiles
    profiles_file = output_path / "character_profiles.json"
    try:
        import json
        profiles_data = {
            name: {
                'gender': profile.gender,
                'age_group': profile.age_group,
                'confidence': profile.confidence,
                'reasoning': profile.reasoning
            }
            for name, profile in character_profiles.items()
        }
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Character profiles saved to {profiles_file}")
    except Exception as e:
        logger.warning(f"Failed to save character profiles: {e}")
    
    # Step 7: Generate TTS audio
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Generating TTS audio")
    logger.info("="*80)
    
    generated_files = []
    for chapter in chapters_list:
        try:
            logger.info(f"\nProcessing chapter {chapter.chapter_number}: {chapter.title}")
            
            chapter_file = tts_engine.generate_chapter_audio(
                chapter=chapter,
                character_profiles=character_profiles,
                character_detector=character_detector,
                voice_mapper=voice_mapper,
                output_dir=output_path
            )
            
            generated_files.append(chapter_file)
            logger.info(f"Chapter {chapter.chapter_number} completed: {chapter_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate chapter {chapter.chapter_number}: {e}")
            logger.warning("Skipping to next chapter...")
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total chapters: {len(chapters_list)}")
    logger.info(f"Successfully generated: {len(generated_files)}")
    logger.info(f"Characters detected: {len(characters)}")
    logger.info(f"Characters analyzed: {len(character_profiles)}")
    logger.info(f"\nOutput directory: {output_path}")
    
    if generated_files:
        logger.info("\nGenerated audio files:")
        for f in generated_files:
            logger.info(f"  - {f}")
    
    logger.info("\n" + "="*80)
    logger.info("Processing complete!")
    logger.info("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='PDF-to-Audiobook Converter - Convert PDF books to audiobooks with character-specific voices'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input PDF file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./output',
        help='Output directory (default: ./output)'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--language', '-l',
        choices=['auto', 'zh', 'en'],
        default='auto',
        help='Language: auto-detect, zh (Chinese), or en (English) (default: auto)'
    )
    
    parser.add_argument(
        '--chapters',
        default='all',
        help='Chapter range to process (e.g., "1-5", "3", or "all" for all chapters) (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Override logging level if verbose
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    
    # Setup logging
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    
    # Check input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Process PDF
    try:
        process_pdf(
            pdf_path=args.input,
            output_dir=args.output,
            config=config,
            language=args.language,
            chapters=args.chapters
        )
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
