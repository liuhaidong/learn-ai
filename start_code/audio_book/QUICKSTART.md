# Quick Start Guide

## 1. Installation Complete ✓

All dependencies are installed. You can now use the audio book converter.

## 2. Setup DeepSeek API Key

### Option A: Set Environment Variable
```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

### Option B: Edit Config File
Edit `config.yaml` and replace:
```yaml
llm:
  api_key: "${DEEPSEEK_API_KEY}"
```
With your actual key:
```yaml
llm:
  api_key: "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## 3. Get a DeepSeek API Key

1. Go to https://platform.deepseek.com/
2. Sign up for an account
3. Go to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

## 4. Basic Usage

```bash
# Convert a PDF book to audiobook
python3 src/main.py --input book.pdf --output ./audiobook
```

### Example Commands

```bash
# Process only first 3 chapters
python3 src/main.py --input book.pdf --output ./audiobook --chapters 1-3

# Specify Chinese language
python3 src/main.py --input book.pdf --output ./audiobook --language zh

# Verbose output for debugging
python3 src/main.py --input book.pdf --output ./audiobook --verbose
```

## 5. What Happens During Processing

The tool will:
1. **Extract text** from your PDF
2. **Detect language** (Chinese or English)
3. **Split into chapters** (by character count)
4. **Find dialogues** and identify characters
5. **Analyze characters** using DeepSeek AI (gender, age)
6. **Generate audio** with character-specific voices
7. **Save MP3 files** to output directory

### First Run Note

On the first run, Coqui TTS will automatically download the XTTS v2 model (several GB). This may take a while depending on your internet connection.

## 6. Output

After processing, you'll find:
```
audiobook/
├── chapter_01.mp3
├── chapter_02.mp3
├── chapter_03.mp3
└── character_profiles.json
```

## 7. Troubleshooting

### "DeepSeek API error"
- Check your API key is correct
- Verify you have API credits in your account
- Check your internet connection

### "No characters detected"
- Your book may not have dialogues
- Edit `dialogue_patterns` in `config.yaml` to match your book's format

### "TTS generation slow"
- First run downloads models (slow)
- Subsequent runs are faster
- Use CUDA GPU if available: set `device: "cuda"` in config.yaml

## 8. Configuration Tips

### Adjust Chapter Length
In `config.yaml`:
```yaml
pdf:
  chapter_length: 2500  # Make chapters shorter/longer
```

### Change Voices
In `config.yaml`, edit `voice_mapping`:
```yaml
voice_mapping:
  adult_male:
    zh: "zh-CN-YunxiNeural"  # Change voice here
```

### Available Voices

**Chinese:**
- `zh-CN-XiaoxiaoNeural` (Female, adult)
- `zh-CN-YunxiNeural` (Male, adult)
- `zh-CN-YunyangNeural` (Male, elderly)
- `zh-CN-XiaoyiNeural` (Female, elderly)

**English:**
- `en-US-JennyNeural` (Female, adult)
- `en-US-GuyNeural` (Male, adult)
- `en-US-TonyNeural` (Male, elderly)
- `en-US-AriaNeural` (Female, elderly)

## 9. Next Steps

- Try with a short sample PDF first
- Review `character_profiles.json` to see character analysis
- Adjust voice mappings to your preference
- Check logs: `cat audiobook.log`

## Need Help?

Check the full documentation in `README.md` or open an issue on GitHub.
