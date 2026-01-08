"""Instagram Automation - Instagram Client Module"""

import random
import asyncio
from typing import Optional, Dict, List
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, ChallengeRequired
from database.models import Account


class InstagramClientWrapper:
    """Wrapper for instagrapi Client with multi-account support"""

    def __init__(self):
        self.clients: Dict[int, Client] = {}
        self.account_configs: Dict[int, Dict] = {}
        self.active_account_ids: List[int] = []

    async def initialize_account(
        self,
        account_id: int,
        username: str,
        password: str,
        proxy: Optional[str] = None
    ) -> Client:
        """Initialize and login Instagram account"""
        try:
            client = Client()
            
            # Set proxy if configured
            if proxy:
                client.set_proxy(proxy)
            
            # Login
            client.login(username, password)
            
            # Cache client and session data
            self.clients[account_id] = client
            self.active_account_ids.append(account_id)
            
            print(f"✅ Account {username} initialized successfully")
            return client
            
        except LoginRequired as e:
            print(f"❌ Login failed for {username}: {e}")
            raise
        except ChallengeRequired as e:
            print(f"❌ Challenge required for {username}: {e}")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize account {username}: {e}")
            raise

    def get_client(self, account_id: int) -> Optional[Client]:
        """Get client by account ID"""
        return self.clients.get(account_id)

    async def get_or_create_client(
        self,
        account_id: int,
        account_config: Dict
    ) -> Client:
        """Get existing client or create new one"""
        if account_id in self.clients:
            return self.clients[account_id]
        
        return await self.initialize_account(
            account_id=account_id,
            username=account_config['username'],
            password=account_config['password'],
            proxy=account_config.get('proxy')
        )

    async def logout_account(self, account_id: int):
        """Logout and cleanup account"""
        if account_id in self.clients:
            try:
                client = self.clients[account_id]
                client.logout()
                del self.clients[account_id]
                if account_id in self.active_account_ids:
                    self.active_account_ids.remove(account_id)
                print(f"✅ Account {account_id} logged out")
            except Exception as e:
                print(f"❌ Failed to logout account {account_id}: {e}")

    async def logout_all(self):
        """Logout all accounts"""
        for account_id in list(self.clients.keys()):
            await self.logout_account(account_id)

    def get_active_accounts(self) -> List[int]:
        """Get list of active account IDs"""
        return self.active_account_ids.copy()

    def is_account_active(self, account_id: int) -> bool:
        """Check if account is active"""
        return account_id in self.active_account_ids


class InstagramActions:
    """Wrapper for Instagram API actions"""

    def __init__(self, client: Client):
        self.client = client

    async def like_post(self, media_id: int) -> bool:
        """Like a post"""
        try:
            self.client.media_like(media_id)
            await asyncio.sleep(random.uniform(5, 15))
            return True
        except Exception as e:
            print(f"❌ Failed to like post {media_id}: {e}")
            return False

    async def comment_on_post(
        self,
        media_id: int,
        comment: str
    ) -> bool:
        """Comment on a post"""
        try:
            self.client.media_comment(media_id, comment)
            await asyncio.sleep(random.uniform(15, 30))
            return True
        except Exception as e:
            print(f"❌ Failed to comment on post {media_id}: {e}")
            return False

    async def follow_user(self, user_id: int) -> bool:
        """Follow a user"""
        try:
            self.client.user_follow(user_id)
            await asyncio.sleep(random.uniform(20, 40))
            return True
        except Exception as e:
            print(f"❌ Failed to follow user {user_id}: {e}")
            return False

    async def upload_photo(
        self,
        photo_path: str,
        caption: str,
        hashtags: List[str] = None
    ) -> Optional[str]:
        """Upload photo post"""
        try:
            # Add hashtags to caption
            if hashtags:
                full_caption = f"{caption}\n\n{' '.join(hashtags)}"
            else:
                full_caption = caption
            
            media = self.client.photo_upload(
                path=photo_path,
                caption=full_caption
            )
            print(f"✅ Photo uploaded successfully")
            return str(media.pk)
        except Exception as e:
            print(f"❌ Failed to upload photo: {e}")
            return None

    async def get_user_info_by_username(self, username: str) -> Optional[Dict]:
        """Get user info by username"""
        try:
            user_info = self.client.user_info_by_username(username)
            return {
                'user_id': user_info.pk,
                'username': user_info.username,
                'full_name': user_info.full_name,
                'biography': user_info.biography,
                'follower_count': user_info.follower_count,
                'following_count': user_info.following_count,
                'is_private': user_info.is_private,
                'profile_pic_url': user_info.profile_pic_url
            }
        except Exception as e:
            print(f"❌ Failed to get user info for {username}: {e}")
            return None

    async def get_user_medias(
        self,
        user_id: int,
        amount: int = 20
    ) -> List:
        """Get user's recent posts"""
        try:
            medias = self.client.user_medias(user_id, amount=amount)
            return medias
        except Exception as e:
            print(f"❌ Failed to get user medias: {e}")
            return []

    async def get_user_followers(
        self,
        user_id: int,
        amount: int = 50
    ) -> List:
        """Get user's followers"""
        try:
            followers = self.client.user_followers(user_id, amount=amount)
            return followers
        except Exception as e:
            print(f"❌ Failed to get user followers: {e}")
            return []

    async def search_hashtag_medias(
        self,
        hashtag: str,
        amount: int = 20
    ) -> List:
        """Get recent media from hashtag"""
        try:
            # Remove # if present
            clean_hashtag = hashtag.lstrip('#')
            medias = self.client.hashtag_medias_recent(clean_hashtag, amount=amount)
            return medias
        except Exception as e:
            print(f"❌ Failed to search hashtag {hashtag}: {e}")
            return []

    async def explore_feed(self, amount: int = 20) -> List:
        """Get explore feed"""
        try:
            medias = self.client.explore_medias(amount=amount)
            return medias
        except Exception as e:
            print(f"❌ Failed to get explore feed: {e}")
            return []
