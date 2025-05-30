# Implementing Keyword Replacement in Streaming LLM Responses

When working with streaming responses from LLM APIs, you may want to replace certain keywords in real-time as the tokens are being generated. Here's how you can implement this:

## Basic Approach

```python
def process_streaming_response(response_stream, replacements):
    """
    Process a streaming LLM response and replace keywords on the fly.
    
    Args:
        response_stream: Iterator of response chunks from the LLM API
        replacements: Dictionary of {keyword: replacement} pairs
    """
    buffer = ""
    
    for chunk in response_stream:
        # Get the new text from this chunk
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            # For OpenAI-style responses
            new_text = chunk.choices[0].delta.content or ""
        else:
            # For simple text chunks
            new_text = chunk
            
        # Add to buffer
        buffer += new_text
        
        # Process buffer for replacements
        for keyword, replacement in replacements.items():
            buffer = buffer.replace(keyword, replacement)
        
        # Yield the processed chunk
        yield buffer[-len(new_text):]
        
        # Keep buffer manageable by trimming what we've already processed
        # but keep enough context for potential partial keyword matches
        max_keyword_length = max([len(k) for k in replacements.keys()], default=0)
        if len(buffer) > max_keyword_length * 2:
            buffer = buffer[-max_keyword_length:]
```

## Improved Implementation

The basic approach has issues with partial keyword matches. Here's a more robust implementation:

```python
def process_streaming_response(response_stream, replacements):
    buffer = ""
    output_index = 0
    
    for chunk in response_stream:
        # Extract text from chunk based on API format
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            new_text = chunk.choices[0].delta.content or ""
        else:
            new_text = chunk
            
        # Add to buffer
        buffer += new_text
        
        # Apply replacements to the entire buffer
        processed_buffer = buffer
        for keyword, replacement in replacements.items():
            processed_buffer = processed_buffer.replace(keyword, replacement)
        
        # Determine what new content to yield
        to_yield = processed_buffer[output_index:]
        if to_yield:
            yield to_yield
            
        # Update output index
        output_index = len(processed_buffer)
```

## Usage Example with OpenAI API

```python
import openai

client = openai.OpenAI()
replacements = {
    "artificial intelligence": "AI",
    "language model": "LM",
    "OpenAI": "ACME Corp"
}

def get_streaming_response():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Explain what ChatGPT is"}],
        stream=True
    )
    
    for processed_chunk in process_streaming_response(response, replacements):
        print(processed_chunk, end="", flush=True)

get_streaming_response()
```

## Handling Edge Cases

For more complex replacements or to handle edge cases:

1. **Cross-chunk keywords**: The buffer approach ensures keywords split across chunks are properly replaced.

2. **Regex replacements**: For pattern-based replacements:

```python
import re

def process_with_regex(response_stream, pattern_replacements):
    buffer = ""
    output_index = 0
    
    for chunk in response_stream:
        # Extract new text
        new_text = extract_text_from_chunk(chunk)
        buffer += new_text
        
        # Apply regex replacements
        processed_buffer = buffer
        for pattern, replacement in pattern_replacements:
            processed_buffer = re.sub(pattern, replacement, processed_buffer)
        
        # Yield new content
        yield processed_buffer[output_index:]
        output_index = len(processed_buffer)
```

3. **Word boundary awareness**:

```python
def replace_with_word_boundaries(text, replacements):
    for keyword, replacement in replacements.items():
        pattern = r'\b' + re.escape(keyword) + r'\b'
        text = re.sub(pattern, replacement, text)
    return text
```

