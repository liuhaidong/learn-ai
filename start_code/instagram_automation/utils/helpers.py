"""Instagram Automation - Utility Functions"""

import random
import asyncio
from datetime import datetime
from typing import List, Dict


def random_delay(min_seconds: float = 5.0, max_seconds: float = 30.0):
    """Random delay with range"""
    return random.uniform(min_seconds, max_seconds)


def format_timestamp(dt: datetime = None) -> str:
    """Format timestamp for logging"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_cost(cost: float) -> str:
    """Format cost for display"""
    return f"${cost:.2f}"


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text for display"""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text"""
    import re
    return re.findall(r'#\w+', text)


def clean_caption(caption: str) -> str:
    """Remove duplicate hashtags and clean up caption"""
    hashtags = extract_hashtags(caption)
    unique_hashtags = list(set(hashtags))
    
    # Remove hashtags from caption
    text_without_hashtags = caption
    for tag in hashtags:
        text_without_hashtags = text_without_hashtags.replace(tag, '')
    
    # Add unique hashtags back
    if unique_hashtags:
        return f"{text_without_hashtags.strip()}\n\n{' '.join(unique_hashtags)}"
    return text_without_hashtags


def select_random_items(items: List, count: int) -> List:
    """Select random items from list"""
    if len(items) <= count:
        return items
    return random.sample(items, count)


def format_percentage(value: float, total: float) -> str:
    """Calculate and format percentage"""
    if total == 0:
        return "0%"
    percentage = (value / total) * 100
    return f"{percentage:.1f}%"


def validate_hashtags(hashtags: List[str]) -> List[str]:
    """Validate hashtags and remove invalid ones"""
    valid_hashtags = []
    
    for tag in hashtags:
        tag = tag.strip()
        if tag and tag.startswith('#') and len(tag) <= 30:
            # Remove special characters (except #)
            cleaned = '#' + ''.join(c for c in tag[1:] if c.isalnum() or c in ['_', '-'])
            if len(cleaned) >= 2:  # Minimum: #x
                valid_hashtags.append(cleaned)
    
    return valid_hashtags


def calculate_engagement_rate(likes: int, comments: int, followers: int) -> float:
    """Calculate engagement rate"""
    if followers == 0:
        return 0.0
    return ((likes + comments * 2) / followers) * 100


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split list into chunks"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Retry function with exponential backoff"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"⚠️  Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
    
    raise last_exception


def generate_product_name(category: str) -> str:
    """Generate a product name for testing"""
    adjectives = {
        'charging_cable': ['Premium', 'Fast', 'Durable', 'Tangle-Free', 'Heavy-Duty'],
        'charger': ['Compact', 'Rapid', 'Universal', 'Smart', 'Portable'],
        'earbuds': ['Pro', 'Wireless', 'Premium', 'Studio', 'Elite'],
        'phone_film': ['Ultra', 'Crystal', 'Privacy', 'Anti-Spy', 'Max-Clear'],
        'phone_case': ['Slim', 'Rugged', 'Designer', 'Clear', 'Protective'],
        'noise_cancelling_headphone': ['Pro', 'Max', 'Elite', 'Studio', 'Audiophile']
    }
    
    adj = random.choice(adjectives.get(category, ['Premium']))
    return f"{adj} {category.replace('_', ' ').title()}"


def calculate_optimal_posting_time() -> str:
    """Calculate optimal posting time based on current time"""
    now = datetime.now()
    hour = now.hour
    
    if 9 <= hour < 11:
        return "morning"
    elif 11 <= hour < 14:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening"
    else:
        return "late_night"


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """Mask sensitive data for logging"""
    if len(data) <= visible_chars:
        return data
    return data[:visible_chars] + '*' * (len(data) - visible_chars)
