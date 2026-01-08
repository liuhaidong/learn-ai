"""Instagram Automation - Content Generation Service for E-commerce"""

import random
import asyncio
from typing import Dict, List, Optional
from ai.openai_client import OpenAIClient
from ai.midjourney_client import MidjourneyClient
from instagram.account_manager import TwoAccountManager
from config.constants import (
    PRODUCT_CATEGORIES,
    CONTENT_TYPE_DISTRIBUTION,
    ECOMMERCE_HASHTAGS,
    CAPTION_TEMPLATES,
    IMAGE_STYLES
)


class ContentGeneratorService:
    """Content generation service for 6 product categories - E-commerce focused"""

    def __init__(self, account_manager: TwoAccountManager):
        self.account_manager = account_manager
        self.openai_client = OpenAIClient()
        self.midjourney_client = MidjourneyClient()

    async def generate_post_content(
        self,
        account_id: int,
        category: str,
        content_type: str = "product_review",
        product_name: str = None,
        product_price: float = None,
        key_features: List[str] = None
    ) -> Dict:
        """Generate complete post content (caption + image)"""

        print(f"\n📝 Generating content for {category} - {content_type}")

        # Get product info from category if not provided
        category_config = PRODUCT_CATEGORIES.get(category, {})
        if not product_name:
            product_name = f"Premium {category_config.get('name', 'Product')}"
        if not product_price:
            price_range = category_config.get('price_range', {})
            product_price = random.uniform(price_range['min'], price_range['max'])
        if not key_features:
            key_features = category_config.get('key_features', [])[:3]

        # 1. Generate caption
        print("📝 Generating caption...")
        caption_result = await self.openai_client.generate_caption_ecommerce(
            product_category=category,
            product_name=product_name,
            product_price=product_price,
            key_features=key_features,
            content_type=content_type,
            language="en"
        )

        caption = caption_result['caption']
        tokens_used = caption_result['tokens_used']

        # 2. Generate hashtags
        print("🏷️  Generating hashtags...")
        hashtags = await self.openai_client.generate_hashtags(
            category=category,
            additional_hashtags=ECOMMERCE_HASHTAGS
        )

        # 3. Generate image
        print("🖼️  Generating product image...")
        image_result = await self.midjourney_client.generate_ecommerce_product_image(
            category=category,
            product_name=product_name,
            style="studio_shot",
            background_color="white"
        )

        image_url = image_result['image_url']
        image_cost = image_result['cost']
        is_cached = image_result.get('cached', False)

        # Combine caption with hashtags
        final_caption = f"{caption}\n\n{' '.join(hashtags)}"

        # Calculate total cost
        caption_cost = tokens_used * 0.00015  # gpt-4o-mini: $0.15/1K tokens
        total_cost = caption_cost + image_cost

        result = {
            'account_id': account_id,
            'category': category,
            'content_type': content_type,
            'product_name': product_name,
            'product_price': product_price,
            'caption': final_caption,
            'image_url': image_url,
            'hashtags': hashtags,
            'cost_tracking': {
                'caption_tokens': tokens_used,
                'caption_cost': caption_cost,
                'image_cost': image_cost,
                'image_cached': is_cached,
                'total_cost': total_cost
            },
            'metadata': {
                'generation_timestamp': asyncio.get_event_loop().time(),
                'model_used': 'gpt-4o-mini',
                'image_style': 'studio_shot'
            }
        }

        print(f"\n✅ Content generated successfully!")
        print(f"   Caption: {caption[:100]}...")
        print(f"   Hashtags: {len(hashtags)} tags")
        print(f"   Image: {'CACHED' if is_cached else 'NEW'}")
        print(f"   Total Cost: ${total_cost:.4f}")

        return result

    async def generate_batch_posts(
        self,
        account_id: int,
        num_posts: int = 3
    ) -> List[Dict]:
        """Generate multiple posts for account"""

        print(f"\n📦 Generating batch of {num_posts} posts for account {account_id}")

        # Get account config to determine categories
        # Account 1: phone_case, phone_film
        # Account 2: earbuds, noise_cancelling_headphone
        # Shared: charging_cable, charger

        posts = []
        categories = self._get_account_categories(account_id)

        for i in range(num_posts):
            # Rotate through categories
            category = categories[i % len(categories)]

            # Select content type
            content_type = self._select_content_type()

            # Generate post
            post_data = await self.generate_post_content(
                account_id=account_id,
                category=category,
                content_type=content_type
            )

            posts.append(post_data)

            # Add delay to avoid rate limiting
            if i < num_posts - 1:
                await asyncio.sleep(5)

        print(f"\n✅ Batch generation complete: {len(posts)} posts")

        return posts

    def _get_account_categories(self, account_id: int) -> List[str]:
        """Get categories for account based on configuration"""
        if account_id == 1:
            return ['phone_case', 'phone_film']
        elif account_id == 2:
            return ['earbuds', 'noise_cancelling_headphone']
        else:
            # Default - all categories
            return list(PRODUCT_CATEGORIES.keys())

    def _select_content_type(self) -> str:
        """Select content type based on distribution"""
        types = list(CONTENT_TYPE_DISTRIBUTION.keys())
        weights = list(CONTENT_TYPE_DISTRIBUTION.values())
        return random.choices(types, weights=weights, k=1)[0]

    async def generate_value_deal_post(
        self,
        account_id: int,
        category: str,
        discount_percent: int = 20
    ) -> Dict:
        """Generate special offer/value deal post"""

        category_config = PRODUCT_CATEGORIES.get(category, {})
        price_range = category_config.get('price_range', {})
        product_name = f"Best-Selling {category_config.get('name', 'Product')}"
        original_price = random.uniform(price_range['min'], price_range['max'])
        deal_price = original_price * (1 - discount_percent / 100)

        content_type = "value_deal"

        system_prompt = f"""You are creating a LIMITED TIME OFFER post for Instagram.

Offer Details:
- Product: {product_name}
- Original Price: ${original_price:.2f}
- Deal Price: ${deal_price:.2f}
- Discount: {discount_percent}% OFF
- Category: {category}

Create urgency and excitement!

CTA Examples:
- "🔥 FLASH SALE - 24 hours only! Link in bio"
- "💥 Limited stock - Don't miss out! Shop now"
- "⚡ 50% OFF - Shop link in bio before it's gone!"
- "🎁 Perfect gift idea - Free shipping! Link in bio"
"""

        user_prompt = f"""Create an Instagram caption for a {discount_percent}% OFF flash sale on {product_name}.

Make it sound EXCITING and create URGENCY!

Structure:
1. Hook with 🔥 emoji and attention-grabbing opening
2. Show the savings clearly ({discount_percent}% OFF from ${original_price:.2f} to ${deal_price:.2f})
3. Highlight scarcity (Limited time, Only X left, etc.)
4. Add urgency (Today only, 24h sale, etc.)
5. Strong CTA with emoji
6. 15-20 relevant hashtags

Keep it under 1800 characters.
Make followers want to buy NOW!
"""

        result = await self.openai_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.9,  # High creativity for offers
            max_tokens=400
        )

        caption = result['text']

        # Generate hashtags
        hashtags = await self.openai_client.generate_hashtags(
            category=category,
            additional_hashtags=ECOMMERCE_HASHTAGS
        )

        # Add urgency hashtags
        urgency_tags = ['#flashsale', '#limitedoffer', '#deals', '#shopnow']
        all_hashtags = hashtags + urgency_tags[:2]

        final_caption = f"{caption}\n\n{' '.join(all_hashtags)}"

        return {
            'account_id': account_id,
            'category': category,
            'content_type': content_type,
            'product_name': product_name,
            'product_price': deal_price,
            'discount_percent': discount_percent,
            'caption': final_caption,
            'cost_tracking': {
                'caption_tokens': result['tokens_used'],
                'caption_cost': result['tokens_used'] * 0.00015,
                'image_cost': 0.0,  # No image for deal posts (reuse existing)
                'total_cost': result['tokens_used'] * 0.00015
            }
        }


class SixCategoryContentStrategy:
    """Content strategy for 6 product categories"""

    CONTENT_ANGLES = {
        "charging_cable": [
            "durability test with bend stress",
            "fast charging speed comparison",
            "tangle-free demonstration",
            "compatibility with multiple devices"
        ],
        "charger": [
            "portability and travel-friendly",
            "fast charging benchmarks",
            "multi-device charging station",
            "safety features showcase"
        ],
        "earbuds": [
            "sound quality in different music genres",
            "noise cancellation test",
            "battery life real-world test",
            "comfort during long wear"
        ],
        "phone_film": [
            "scratch resistance test",
            "installation ease demonstration",
            "touch sensitivity comparison",
            "clarity before/after application"
        ],
        "phone_case": [
            "drop protection test",
            "design aesthetics showcase",
            "wireless charging compatibility",
            "color and material options"
        ],
        "noise_cancelling_headphone": [
            "ANC effectiveness in different environments",
            "transparency mode comparison",
            "battery life with ANC on/off",
            "comfort for long flights"
        ]
    }

    @staticmethod
    def get_content_idea(category: str, previous_ideas: List[str] = None) -> str:
        """Get content idea for category"""
        if previous_ideas and len(previous_ideas) >= 10:
            # Cycle back to first idea after 10 posts
            return previous_ideas[0]

        angles = SixCategoryContentStrategy.CONTENT_ANGLES.get(category, [])
        return random.choice(angles) if angles else "product showcase"
