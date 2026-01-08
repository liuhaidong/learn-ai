# PDF-to-Audiobook Converter MVP

Converts PDF books to audiobooks with character-specific voices using DeepSeek AI and Coqui TTS.

## Features

- **PDF Text Extraction**: Extract text from PDF files with automatic cleaning
- **Chapter Splitting**: Split books into chapters by character count
- **Character Detection**: Identify characters and their dialogues
- **AI Character Analysis**: Use DeepSeek API to analyze character gender and age
- **Multi-Voice TTS**: Generate speech with character-appropriate voices
- **Language Support**: Chinese and English with automatic detection
- **Error Handling**: Graceful degradation - skips errors and continues processing

## Requirements

- Python 3.10+
- DeepSeek API key
- FFmpeg (for audio processing)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### 3. Configure DeepSeek API Key

Edit `config.yaml` and set your DeepSeek API key:

```yaml
llm:
  api_key: "your_deepseek_api_key_here"
```

Or set environment variable:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

## Usage

### Basic Usage

```bash
python src/main.py --input book.pdf --output ./audiobook
```

### Advanced Options

```bash
python src/main.py \
  --input book.pdf \
  --output ./audiobook \
  --config config.yaml \
  --language auto \
  --chapters all \
  --verbose
```

### Arguments

- `--input`, `-i`: Input PDF file path (required)
- `--output`, `-o`: Output directory (default: `./output`)
- `--config`, `-c`: Configuration file (default: `config.yaml`)
- `--language`, `-l`: Language detection - `auto`, `zh` (Chinese), or `en` (English) (default: `auto`)
- `--chapters`: Chapter range to process - `1-5`, `3`, or `all` (default: `all`)
- `--verbose`, `-v`: Enable verbose output

## Configuration

Edit `config.yaml` to customize behavior:

### PDF Processing

```yaml
pdf:
  chapter_length: 2500  # characters per chapter
  min_chapter_length: 1500
  max_chapter_length: 4000
```

### Character Detection

```yaml
character_detection:
  dialogue_patterns:
    - '"(.*?)"'       # English quotes
    - '「(.*?)」'      # Chinese quotes
    - "'(.*?)'"        # English single quotes
  min_dialogue_length: 5
```

### TTS Settings

```yaml
tts:
  model: "tts_models/multilingual/multi-dataset/xtts_v2"
  device: "cpu"  # or "cuda"
  voice_mapping:
    adult_male:
      zh: "zh-CN-YunxiNeural"
      en: "en-US-GuyNeural"
```

### Output Settings

```yaml
output:
  format: "mp3"
  bitrate: "128k"
  chapter_padding: 0.5  # seconds
```

## Output Structure

After processing, output directory will contain:

```
output/
├── chapter_01.mp3
├── chapter_02.mp3
├── chapter_03.mp3
└── character_profiles.json
```

## Voice Mapping

The system automatically selects voices based on character gender and age:

- **Adult Male**: Deep voice
- **Adult Female**: Medium voice (default)
- **Young Adult**: Standard voice
- **Elderly**: Lower pitch, slightly slower
- **Child**: Higher pitch

## Processing Pipeline

1. **Extract Text**: Read PDF and extract clean text
2. **Detect Language**: Auto-detect Chinese or English
3. **Split Chapters**: Divide text into chapters by character count
4. **Detect Dialogues**: Extract all dialogues with speaker context
5. **Build Characters**: Create character database from dialogues
6. **AI Analysis**: Use DeepSeek to determine gender and age
7. **Generate TTS**: Create audio with appropriate voices
8. **Concatenate**: Merge segments with silence padding

## Troubleshooting

### "Cannot find FFmpeg"

Install FFmpeg (see Installation section) and ensure it's in your PATH.

### "Failed to initialize TTS"

Coqui TTS will auto-download models on first run. Ensure you have:
- Stable internet connection
- 5+ GB free disk space
- Python 3.10+

### "DeepSeek API error"

Check:
- Your API key is correct in `config.yaml`
- Your API account has credits
- Network connection is working

### "No characters detected"

This happens if:
- The PDF has no dialogues
- Dialogue patterns don't match your book's format

Edit `dialogue_patterns` in `config.yaml` to match your book's format.

### Poor voice quality

- Ensure you're using the XTTS v2 model (default)
- Check if device is set to "cuda" if you have GPU
- Adjust `voice_parameters` in `config.yaml`

## Limitations (MVP)

- Simple chapter splitting by character count (not by actual chapter headings)
- Basic speaker identification (relies on context patterns)
- No voice cloning (uses preset Coqui voices)
- Sequential processing (no parallelization)
- English and Chinese only

## Future Enhancements

- [ ] Smart chapter detection from PDF bookmarks
- [ ] Advanced speaker identification using NLP
- [ ] Voice cloning for specific characters
- [ ] Parallel processing for faster generation
- [ ] Support for more languages
- [ ] Web UI for easier interaction
- [ ] Batch processing of multiple books

## Contributing

This is an MVP project. Issues and pull requests welcome!

## License

MIT License

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
