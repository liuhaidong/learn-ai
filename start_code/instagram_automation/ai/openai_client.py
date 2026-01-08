"""Instagram Automation - OpenAI Client Integration"""

import openai
from typing import Optional, Dict, Any
from config.settings import settings


class OpenAIClient:
    """OpenAI API Client for text generation"""

    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key
        )
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature
            )

            result = response.choices[0].message.content

            # Return with metadata for cost tracking
            return {
                'text': result,
                'tokens_used': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }

        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            raise

    async def generate_caption_ecommerce(
        self,
        product_category: str,
        product_name: str,
        product_price: float,
        key_features: list,
        content_type: str = "product_review",
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate Instagram caption for e-commerce product"""

        system_prompt = f"""You are an expert Instagram content creator for {language}-language e-commerce cross-border sales focused on phone accessories and tech gadgets.

Your role: Create engaging, conversion-oriented Instagram captions that drive sales.

Tone and Style:
- Professional yet approachable and friendly
- Sales-focused but not pushy
- Highlight value and benefits, not just features
- Use emojis strategically to increase engagement
- Create urgency without being aggressive

Key Elements to Include:
1. Hook: Grab attention in first 1-2 lines
2. Benefits: Focus on what the product DOES for the customer
3. Social Proof: Mention "bestseller", "customer favorite", "limited stock" when appropriate
4. CTA: Strong call-to-action (Shop link in bio, Limited time offer, Use discount code)
5. Hashtags: 15-25 relevant tags for visibility

Format Guidelines:
- 1500-2200 characters
- Use line breaks for readability
- Include 4-7 relevant emojis
- End with question or CTA to encourage engagement
- Include at least 2-3 hashtags from provided list

E-commerce Best Practices:
- Focus on "value for money", "quality", "durability"
- Use scarcity: "Limited stock", "Only X left", "24h flash sale"
- Include shipping info: "Free worldwide shipping", "Fast delivery"
- Target pain points and provide solutions
"""

        user_prompt = f"""Create an Instagram caption for a {product_name} (a {product_category})

Product Details:
- Price: ${product_price:.2f}
- Key Features: {', '.join(key_features)}
- Content Type: {content_type}

Requirements:
1. Create an attention-grabbing hook with emoji
2. Highlight 3-4 main benefits/value propositions
3. Include at least one use case scenario
4. Add urgency/scarcity element (limited stock, special offer, etc.)
5. End with strong CTA (e.g., "Link in bio - Limited stock!", "Shop now for 10% OFF")
6. Keep between 1500-2200 characters
7. Use appropriate emojis (4-7 total)
8. Make it sound authentic and engaging

Target audience: Tech-savvy consumers looking for quality accessories at great prices.
"""

        result = await self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.8  # Higher temperature for more creative captions
        )

        return {
            'caption': result['text'],
            'tokens_used': result['tokens_used'],
            'metadata': {
                'product_category': product_category,
                'product_name': product_name,
                'product_price': product_price,
                'content_type': content_type,
                'language': language
            }
        }

    async def generate_comment_ecommerce(
        self,
        category: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """Generate engaging comments for e-commerce posts"""

        system_prompt = f"""You are an authentic Instagram user who comments on tech accessory posts.

Your personality: Helpful, enthusiastic, and genuinely interested in tech products.

Comment Guidelines:
- Be authentic and conversational, not spammy
- Ask relevant questions or share experiences
- Keep under 3 sentences, ideally 1-2 sentences
- Use 1-2 emojis maximum
- Avoid generic phrases like "nice post" or "great job"
- Show interest in the product or ask about details
"""

        user_prompt = f"""Write a natural Instagram comment for a {category} post.

Context: {context if context else 'No specific context, general tech interest'}

Comment style:
- Authentic and conversational
- 1-2 sentences maximum
- 1-2 emojis
- Sound like a real person, not a bot
- Ask about the product or share a relevant thought

Make it sound like a genuine tech enthusiast comment.
"""

        result = await self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.9,  # Higher temperature for more variety
            max_tokens=100
        )

        return {
            'comment': result['text'],
            'tokens_used': result['tokens_used']
        }

    async def generate_hashtags(
        self,
        category: str,
        additional_hashtags: list = []
    ) -> list:
        """Generate relevant hashtags for a product category"""

        system_prompt = f"""You are an Instagram hashtag expert for e-commerce and tech products.

Hashtag Strategy:
- Mix of popular, niche-specific, and long-tail tags
- 15-25 total hashtags
- Include location-based tags when appropriate
- Avoid banned or overused tags

Categories to consider:
- Popular/General: High volume but high competition
- Niche-Specific: Lower volume, higher relevance
- Long-tail: Very low volume, highly targeted
- E-commerce: Shopping, deal, urgency tags"""

        user_prompt = f"""Generate 15-25 relevant Instagram hashtags for a {category} product.

Additional context/tags to include: {', '.join(additional_hashtags) if additional_hashtags else 'None'}

Hashtag distribution strategy:
- 5-7 popular high-volume tags
- 5-8 niche-specific tags
- 5-10 long-tail tags
- 2-3 urgency/shopping tags (e.g., #shopnow, #deals, #limitedoffer)

Make sure all tags are:
- Relevant to the category
- Properly formatted (starting with #)
- Lowercase or capitalized appropriately
- Not banned/spammy

Return the hashtags as a space-separated string.
"""

        result = await self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.6,
            max_tokens=150
        )

        # Parse hashtags from response
        hashtags = [
            tag.strip() for tag in result['text'].split()
            if tag.strip().startswith('#')
        ]

        return hashtags
