"""Instagram Automation - Midjourney Client Integration"""

import httpx
import asyncio
from typing import Optional, Dict, Any
from config.settings import settings


class MidjourneyClient:
    """Midjourney API Client for image generation (via ImaginePro)"""

    def __init__(self):
        self.base_url = settings.midjourney_base_url
        self.api_key = settings.midjourney_api_key
        self.image_cache: Dict[str, str] = {}
        self.enable_cache = settings.enable_image_cache

    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "4:5",
        quality: str = "hd",
        wait_timeout: int = 600
    ) -> Dict[str, Any]:
        """Generate image using Midjourney API via ImaginePro"""

        # Check cache
        cache_key = f"{prompt}:{aspect_ratio}:{quality}"
        if self.enable_cache and cache_key in self.image_cache:
            print(f"📦 Using cached image for: {prompt[:50]}...")
            return {
                'image_url': self.image_cache[cache_key],
                'cached': True,
                'cost': 0.0
            }

        try:
            async with httpx.AsyncClient(timeout=900.0) as client:
                # Generate image
                response = await client.post(
                    f"{self.base_url}/nova/imagine",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "prompt": prompt,
                        "ref": f"insta_{asyncio.get_event_loop().time():.timestamp()}",
                        "timeout": wait_timeout
                    }
                )
                response.raise_for_status()

                data = response.json()
                task_id = data.get("data", {}).get("task_id")

                if not task_id:
                    raise Exception("No task ID returned from API")

                # Wait for completion
                image_url = await self._wait_for_completion(task_id, client)

                # Cache the result
                if self.enable_cache:
                    self.image_cache[cache_key] = image_url

                # Estimate cost (standard = $0.05, HD = $0.10)
                cost = 0.05 if quality == "standard" else 0.10

                print(f"✅ Image generated successfully: {image_url}")

                return {
                    'image_url': image_url,
                    'cached': False,
                    'task_id': task_id,
                    'cost': cost,
                    'metadata': {
                        'prompt': prompt,
                        'aspect_ratio': aspect_ratio,
                        'quality': quality
                    }
                }

        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error from Midjourney API: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            raise
        except Exception as e:
            print(f"❌ Midjourney API error: {e}")
            raise

    async def _wait_for_completion(
        self,
        task_id: str,
        client: httpx.AsyncClient,
        max_wait: int = 600,
        check_interval: int = 10
    ) -> str:
        """Wait for image generation to complete"""

        waited = 0
        while waited < max_wait:
            try:
                response = await client.get(
                    f"{self.base_url}/nova/status/{task_id}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}"
                    }
                )
                response.raise_for_status()

                data = response.json()
                status = data.get("data", {}).get("status")

                if status == "completed":
                    image_urls = data.get("data", {}).get("image_urls", [])
                    if image_urls:
                        return image_urls[0]
                    raise Exception("No image URLs in completed response")

                elif status == "failed":
                    error_msg = data.get("data", {}).get("error", "Unknown error")
                    raise Exception(f"Image generation failed: {error_msg}")

                elif status in ["pending", "processing"]:
                    # Still processing, wait and retry
                    print(f"⏳ Image generation in progress... ({waited}s elapsed)")
                    await asyncio.sleep(check_interval)
                    waited += check_interval
                else:
                    print(f"⚠️ Unknown status: {status}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise Exception(f"Task {task_id} not found")
                elif e.response.status_code == 429:
                    print("⚠️ Rate limited, waiting longer...")
                    await asyncio.sleep(30)
                else:
                    raise
            except Exception as e:
                print(f"⚠️ Error checking status: {e}")
                await asyncio.sleep(check_interval)
                waited += check_interval

        raise Exception(f"Image generation timed out after {max_wait}s")

    async def generate_ecommerce_product_image(
        self,
        category: str,
        product_name: str,
        style: str = "studio_shot",
        background_color: str = "white"
    ) -> Dict[str, Any]:
        """Generate e-commerce product image optimized for Instagram"""

        # Build effective prompt for e-commerce
        prompts = {
            "studio_shot": f"""Professional product photography of {product_name}, a {category}
Clean minimalist studio setup, {background_color} background, soft studio lighting
Show product elegantly displayed, centered composition, slight elevation angle
Emphasize: premium build quality, attractive design, professional appearance
Style: ultra-detailed, 4K, photorealistic, high contrast
--ar 4:5 --style raw --v 6.0 --q 2""",

            "lifestyle": f"""Modern lifestyle product photography of {product_name}, a {category}
Clean modern setting, natural lighting, product in authentic use scenario
Show product being used naturally, genuine context, relatable moment
Style: ultra-detailed, 4K, photorealistic, warm tones
--ar 4:5 --style raw --v 6.0 --q 2""",

            "unboxing": f"""Clean unboxing setup of {product_name}, a {category}
Professional unboxing presentation on clean table or desk, product still in packaging
Focus on anticipation and excitement of new product reveal
Style: ultra-detailed, 4K, photorealistic, bright lighting
--ar 4:5 --style raw --v 6.0 --q 2""",

            "comparison": f"""Side-by-side product comparison of {product_name}, a {category}
Clean neutral background, professional comparison setup
Both products evenly lit, same scale, clear differences visible
Style: ultra-detailed, 4K, photorealistic, studio quality
--ar 1:1 --style raw --v 6.0 --q 2"""
        }

        prompt = prompts.get(style, prompts["studio_shot"])

        result = await self.generate_image(
            prompt=prompt,
            aspect_ratio="4:5",  # Instagram portrait
            quality="hd"
        )

        return result

    def clear_cache(self):
        """Clear image cache"""
        self.image_cache.clear()
        print("🗑️ Image cache cleared")

    def get_cache_size(self) -> int:
        """Get current cache size"""
        return len(self.image_cache)
