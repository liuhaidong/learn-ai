"""
Example usage of the PDF-to-Audiobook converter
"""

# Example 1: Basic usage
# python3 src/main.py --input example.pdf --output ./audiobook

# Example 2: Process specific chapters
# python3 src/main.py --input example.pdf --output ./audiobook --chapters 1-3

# Example 3: Specify language
# python3 src/main.py --input example.pdf --output ./audiobook --language zh

# Example 4: Verbose output for debugging
# python3 src/main.py --input example.pdf --output ./audiobook --verbose

# Example 5: Custom config
# python3 src/main.py --input example.pdf --output ./audiobook --config config_test.yaml

print("""
PDF-to-Audiobook Converter Examples
====================================

Basic Usage:
  python3 src/main.py --input book.pdf --output ./audiobook

Advanced Usage:
  python3 src/main.py \\
    --input book.pdf \\
    --output ./audiobook \\
    --config config.yaml \\
    --language auto \\
    --chapters all \\
    --verbose

Process Specific Chapters:
  # First 3 chapters
  python3 src/main.py --input book.pdf --output ./audiobook --chapters 1-3
  
  # Only chapter 5
  python3 src/main.py --input book.pdf --output ./audiobook --chapters 5

Language Options:
  # Auto-detect language
  python3 src/main.py --input book.pdf --output ./audiobook --language auto
  
  # Force Chinese
  python3 src/main.py --input book.pdf --output ./audiobook --language zh
  
  # Force English
  python3 src/main.py --input book.pdf --output ./audiobook --language en

Full Command Example:
  python3 src/main.py \\
    --input my_novel.pdf \\
    --output ./audiobook_output \\
    --config config.yaml \\
    --language auto \\
    --chapters 1-10 \\
    --verbose

Configuration:
  Edit config.yaml to customize:
  - Chapter length
  - Dialogue patterns
  - Voice mappings
  - TTS parameters
  - Output settings

Output:
  The tool will create:
  - chapter_XX.mp3 files
  - character_profiles.json
  - audiobook.log (if configured)

Requirements:
  - DeepSeek API key (set DEEPSEEK_API_KEY environment variable)
  - Python 3.10+
  - FFmpeg (for audio processing)
  - Stable internet connection (for TTS model download on first run)

Notes:
  - First run will download XTTS v2 model (several GB)
  - Processing speed depends on your CPU/GPU
  - Character analysis uses DeepSeek API (requires credits)
  - Errors are skipped gracefully (check audiobook.log for details)
""")
