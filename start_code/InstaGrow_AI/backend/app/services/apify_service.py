from apify_client import ApifyClient
from typing import List, Optional, Dict
from app.core.config import settings

client = ApifyClient(settings.APIFY_API_TOKEN)


class ApifyService:
    @staticmethod
    async def get_instagram_profile(username: str) -> Optional[Dict]:
        try:
            run_input = {
                "directUrls": [f"https://www.instagram.com/{username}/"],
                "resultsType": "posts",
                "resultsLimit": 50,
            }
            
            run = client.actor("apify/instagram-scraper").call(run_input=run_input)
            
            results = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                results.append(item)
            
            if not results:
                return None
            
            profile_data = results[0] if results else {}
            
            return {
                "username": profile_data.get("username", username),
                "full_name": profile_data.get("fullName", ""),
                "bio": profile_data.get("biography", ""),
                "followers": profile_data.get("followersCount", 0),
                "following": profile_data.get("followsCount", 0),
                "posts_count": profile_data.get("postsCount", 0),
                "is_verified": profile_data.get("verified", False),
                "profile_pic_url": profile_data.get("profilePicUrl", ""),
                "external_url": profile_data.get("externalUrl", ""),
                "recent_posts": [
                    {
                        "id": post.get("id", ""),
                        "caption": post.get("caption", ""),
                        "type": post.get("type", "image"),
                        "url": post.get("url", ""),
                        "likes": post.get("likesCount", 0),
                        "comments": post.get("commentsCount", 0),
                        "timestamp": post.get("timestamp", ""),
                        "image_url": post.get("displayUrl", ""),
                        "hashtags": post.get("hashtags", []),
                    }
                    for post in results[:20]
                ],
            }
        except Exception as e:
            raise Exception(f"Failed to get Instagram profile: {str(e)}")

    @staticmethod
    async def search_competitors(
        keywords: List[str],
        hashtags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        competitors = []
        
        try:
            for keyword in keywords:
                run_input = {
                    "search": keyword,
                    "resultsType": "profiles",
                    "resultsLimit": limit,
                }
                
                run = client.actor("apify/instagram-scraper").call(run_input=run_input)
                
                for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                    if len(competitors) >= limit:
                        break
                    
                    competitors.append({
                        "username": item.get("username", ""),
                        "full_name": item.get("fullName", ""),
                        "bio": item.get("biography", ""),
                        "followers": item.get("followersCount", 0),
                        "following": item.get("followsCount", 0),
                        "posts_count": item.get("postsCount", 0),
                    })
            
            return competitors[:limit]
        except Exception as e:
            raise Exception(f"Failed to search competitors: {str(e)}")

    @staticmethod
    async def get_post_comments(post_url: str, limit: int = 50) -> List[Dict]:
        try:
            run_input = {
                "directUrls": [post_url],
                "resultsType": "comments",
                "resultsLimit": limit,
            }
            
            run = client.actor("apify/instagram-scraper").call(run_input=run_input)
            
            comments = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                comments.append({
                    "username": item.get("ownerUsername", ""),
                    "text": item.get("text", ""),
                    "timestamp": item.get("timestamp", ""),
                    "likes": item.get("likesCount", 0),
                })
            
            return comments
        except Exception as e:
            raise Exception(f"Failed to get post comments: {str(e)}")

    @staticmethod
    async def get_followers(username: str, limit: int = 100) -> List[Dict]:
        try:
            run_input = {
                "directUrls": [f"https://www.instagram.com/{username}/"],
                "resultsType": "followers",
                "resultsLimit": limit,
            }
            
            run = client.actor("apify/instagram-scraper").call(run_input=run_input)
            
            followers = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                followers.append({
                    "username": item.get("username", ""),
                    "full_name": item.get("fullName", ""),
                    "profile_pic_url": item.get("profilePicUrl", ""),
                    "is_private": item.get("isPrivate", False),
                })
            
            return followers
        except Exception as e:
            raise Exception(f"Failed to get followers: {str(e)}")

    @staticmethod
    async def like_post(post_url: str, instagram_account: str) -> bool:
        raise NotImplementedError("This feature requires Instagram API credentials and OAuth integration")

    @staticmethod
    async def follow_user(username: str, instagram_account: str) -> bool:
        raise NotImplementedError("This feature requires Instagram API credentials and OAuth integration")

    @staticmethod
    async def comment_on_post(post_url: str, comment: str, instagram_account: str) -> bool:
        raise NotImplementedError("This feature requires Instagram API credentials and OAuth integration")


apify_service = ApifyService()
