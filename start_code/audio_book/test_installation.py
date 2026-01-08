"""
Simple test script to verify installation and basic functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    modules = [
        'yaml',
        'langdetect',
        'openai',
        'torch',
        'pdfplumber',
        'pydub',
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Try TTS separately
    try:
        from TTS.api import TTS
        print("  ✓ TTS (coqui-tts)")
    except ImportError as e:
        print(f"  ✗ TTS - {e}")
        print("Run: pip install coqui-tts")
        failed.append('TTS')
    
    if failed:
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_config():
    """Test if config file exists and is valid"""
    print("\nTesting configuration...")
    
    config_files = ['config.yaml', 'config_test.yaml']
    found = []
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✓ Found {config_file}")
            found.append(config_file)
        else:
            print(f"  ✗ Missing {config_file}")
    
    if not found:
        print("\n❌ No configuration file found!")
        return False
    
    return True


def test_ffmpeg():
    """Test if FFmpeg is installed"""
    print("\nTesting FFmpeg...")
    
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ FFmpeg is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("  ✗ FFmpeg is not installed")
    print("  Install FFmpeg: https://ffmpeg.org/download.html")
    return False


def test_env_vars():
    """Test if required environment variables are set"""
    print("\nTesting environment variables...")
    
    import os
    
    # Check DEEPSEEK_API_KEY
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key and api_key.strip() and api_key != '${DEEPSEEK_API_KEY}':
        print("  ✓ DEEPSEEK_API_KEY is set")
        # Don't print the actual key for security
        print(f"    (length: {len(api_key)} characters)")
        return True
    else:
        print("  ✗ DEEPSEEK_API_KEY is not set")
        print("  Set it with: export DEEPSEEK_API_KEY='your-key-here'")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("Audio Book Converter - Installation Test")
    print("="*80)
    
    results = {
        'imports': test_imports(),
        'config': test_config(),
        'ffmpeg': test_ffmpeg(),
        'env': test_env_vars(),
    }
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed! You're ready to use the audio book converter.")
        print("\nUsage:")
        print("  python src/main.py --input book.pdf --output ./audiobook")
    else:
        print("✗ Some tests failed. Please fix the issues above before continuing.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nTo install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        print("\nTo set API key:")
        print("  export DEEPSEEK_API_KEY='your-key-here'")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
