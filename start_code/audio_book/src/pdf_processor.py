"""
PDF Processing Module
Handles PDF text extraction and chapter splitting
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import pdfplumber
from langdetect import detect


@dataclass
class Chapter:
    """Represents a chapter in the book"""
    title: str
    content: str
    page_range: tuple
    chapter_number: int


class PDFProcessor:
    """Processes PDF files to extract text and split into chapters"""
    
    def __init__(self, config: dict):
        """
        Initialize PDF processor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.chapter_length = config['pdf']['chapter_length']
        self.min_chapter_length = config['pdf']['min_chapter_length']
        self.max_chapter_length = config['pdf']['max_chapter_length']
        self.preserve_formatting = config['pdf']['preserve_formatting']
        
        self.logger = logging.getLogger(__name__)
        
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Full text as string
        """
        self.logger.info(f"Extracting text from {pdf_path}")
        
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            text_parts.append(text)
                        self.logger.debug(f"Extracted page {page_num}: {len(text) if text else 0} chars")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {page_num}: {e}")
                        continue
            
            full_text = '\n'.join(text_parts)
            
            # Clean up text if not preserving formatting
            if not self.preserve_formatting:
                full_text = self._clean_text(full_text)
            
            self.logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up extracted text for better TTS processing
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and other artifacts
        text = re.sub(r'\n+\s*\d+\s*\n+', '\n', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', text)
        text = re.sub(r'([一-龯])-\n([一-龯])', r'\1\2', text)
        
        return text.strip()
    
    def split_by_chapters(self, text: str) -> List[Chapter]:
        """
        Split text into chapters by character count (MVP approach)
        
        Args:
            text: Full text to split
            
        Returns:
            List of Chapter objects
        """
        self.logger.info("Splitting text into chapters")
        
        chapters = []
        chapter_num = 1
        
        # Simple approach: split by character count
        start_idx = 0
        while start_idx < len(text):
            # Calculate end index for this chapter
            end_idx = start_idx + self.chapter_length
            
            # Adjust to break at sentence boundary if possible
            if end_idx < len(text):
                # Look for sentence ending punctuation
                sentence_endings = ['。', '！', '？', '.', '!', '?']
                best_break = None
                
                # Search backwards for a good break point
                for offset in range(200):  # Search up to 200 characters
                    if end_idx + offset < len(text):
                        if text[end_idx + offset] in sentence_endings:
                            best_break = end_idx + offset + 1
                            break
                    if end_idx - offset > start_idx:
                        if text[end_idx - offset] in sentence_endings:
                            best_break = end_idx - offset + 1
                            break
                
                if best_break:
                    end_idx = best_break
            
            # Extract chapter content
            chapter_content = text[start_idx:end_idx].strip()
            
            # Ensure minimum length
            if len(chapter_content) >= self.min_chapter_length:
                # Extract potential title from first sentence
                title = self._extract_chapter_title(chapter_content)
                
                chapter = Chapter(
                    title=title,
                    content=chapter_content,
                    page_range=(chapter_num, chapter_num),  # Simplified for MVP
                    chapter_number=chapter_num
                )
                chapters.append(chapter)
                
                self.logger.info(f"Created chapter {chapter_num}: '{title}' ({len(chapter_content)} chars)")
                chapter_num += 1
            
            start_idx = end_idx
        
        self.logger.info(f"Successfully split text into {len(chapters)} chapters")
        return chapters
    
    def _extract_chapter_title(self, text: str) -> str:
        """
        Extract title from chapter content
        Look for first sentence or line
        
        Args:
            text: Chapter content
            
        Returns:
            Title string
        """
        # Try to extract first sentence/line
        lines = text.split('\n')
        
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and len(line) < 100:  # Reasonable title length
                return line
        
        # Fallback: use first 50 characters
        return text[:50] + ('...' if len(text) > 50 else '')
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Chinese or English
        
        Args:
            text: Text to analyze
            
        Returns:
            "zh" or "en"
        """
        self.logger.debug("Detecting language")
        
        # Use langdetect for accurate detection
        try:
            # Use first 1000 characters for detection
            sample = text[:1000]
            detected_lang = detect(sample)
            
            # Map to our language codes
            if detected_lang in ['zh-cn', 'zh-tw', 'zh']:
                self.logger.info("Detected language: Chinese")
                return "zh"
            else:
                self.logger.info(f"Detected language: {detected_lang} (mapped to English)")
                return "en"
                
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}, defaulting to English")
            return "en"
