from openai import OpenAI, OpenAIError
from typing import List, Optional
from app.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class OpenAIService:
    @staticmethod
    async def generate_caption(
        product_description: str,
        brand_tone: str = "professional",
        target_audience: Optional[str] = None,
        hashtags: bool = True,
    ) -> dict:
        prompt = f"""
        Generate an Instagram caption for the following product:
        
        Product Description: {product_description}
        Brand Tone: {brand_tone}
        Target Audience: {target_audience or 'General audience'}
        
        Requirements:
        1. Create an engaging hook in the first line
        2. Highlight key benefits of the product
        3. Include a clear call-to-action
        4. Use appropriate emojis
        5. Keep it concise but impactful
        
        Format the response as JSON with these keys:
        - "caption": The full caption text
        - "hashtags": List of 10-15 relevant hashtags (mix of popular and niche)
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Instagram marketing copywriter."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            
            import json
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No content returned from OpenAI")
            result = json.loads(content)
            return result
        except OpenAIError as e:
            raise Exception(f"Failed to generate caption: {str(e)}")

    @staticmethod
    async def generate_competitor_analysis(
        competitor_data: dict,
    ) -> dict:
        prompt = f"""
        Analyze the following Instagram competitor data and provide strategic insights:
        
        Competitor Profile: {competitor_data.get('bio', 'N/A')}
        Follower Count: {competitor_data.get('followers', 0)}
        Recent Posts: {len(competitor_data.get('posts', []))}
        
        Content Analysis:
        {competitor_data.get('posts', [])[:10]}
        
        Provide analysis in the following areas:
        1. Value Proposition: What makes this brand unique?
        2. Content Strategy: What types of content perform best?
        3. Posting Pattern: Optimal posting times and frequency
        4. Hashtag Strategy: Effective hashtags they use
        5. Engagement Tactics: How they engage with their audience
        
        Format as JSON with these keys:
        - "value_proposition": string
        - "content_strategy": dict with percentages for post types
        - "best_posting_times": list of time slots
        - "top_hashtags": list of 15 effective hashtags
        - "engagement_rate": float
        - "recommendations": list of actionable suggestions
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Instagram marketing analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            
            import json
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No content returned from OpenAI")
            result = json.loads(content)
            return result
        except OpenAIError as e:
            raise Exception(f"Failed to generate competitor analysis: {str(e)}")

    @staticmethod
    async def generate_comment(
        post_image_url: str,
        post_caption: str,
        context: Optional[str] = None,
    ) -> str:
        prompt = f"""
        Generate a genuine, non-generic Instagram comment for this post.
        
        Post Caption: {post_caption}
        Context: {context or 'General appreciation'}
        
        Requirements:
        1. Be specific and show you actually looked at the content
        2. Ask a relevant follow-up question
        3. Use appropriate emojis
        4. Keep it friendly and authentic
        5. Maximum 2 sentences
        
        Avoid generic phrases like "Nice pic!" or "Love it!"
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a genuine Instagram user who engages authentically."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=100,
            )
            
            comment = response.choices[0].message.content
            if comment is None:
                raise Exception("No content returned from OpenAI")
            return comment.strip()
        except OpenAIError as e:
            raise Exception(f"Failed to generate comment: {str(e)}")

    @staticmethod
    async def generate_image(
        prompt: str,
        style: str = "professional",
        aspect_ratio: str = "1:1",
    ) -> str:
        enhanced_prompt = f"""
        Create a {style} Instagram image: {prompt}
        
        Style guidelines:
        - High quality, professional photography look
        - Clean, modern aesthetic
        - Product-focused with good lighting
        - Minimal distractions
        - Consistent brand feel
        """
        
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024" if aspect_ratio == "1:1" else "1792x1024",
                quality="standard",
                n=1,
            )
            
            if not response.data:
                raise Exception("No images generated")
            
            image_url = response.data[0].url
            if image_url is None:
                raise Exception("No image URL returned")
            return image_url
        except OpenAIError as e:
            raise Exception(f"Failed to generate image: {str(e)}")

    @staticmethod
    async def generate_content_calendar(
        product_category: str,
        brand_tone: str,
        days: int = 7,
    ) -> List[dict]:
        prompt = f"""
        Generate a content calendar for Instagram for the next {days} days.
        
        Product Category: {product_category}
        Brand Tone: {brand_tone}
        
        For each day, provide:
        1. Content type (POST, REEL, STORY)
        2. Post topic/theme
        3. Brief content idea
        4. Suggested posting time
        
        Mix content types to keep engagement high.
        Include a variety of themes: product showcases, behind-the-scenes,
        user-generated content, educational content, and promotional posts.
        
        Format as JSON with a "calendar" key containing an array of daily entries.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Instagram content strategist."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                response_format={"type": "json_object"},
            )
            
            import json
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No content returned from OpenAI")
            result = json.loads(content)
            return result.get("calendar", [])
        except OpenAIError as e:
            raise Exception(f"Failed to generate content calendar: {str(e)}")


openai_service = OpenAIService()
