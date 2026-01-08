"""Instagram Automation - Multi-Account Manager"""

import random
from typing import Dict, Optional
from instagram.client import InstagramClientWrapper, InstagramActions
from database.models import Account
from database.repositories import AccountRepository
from config.constants import ACCOUNT_CONFIGS, SHARED_CATEGORIES


class TwoAccountManager:
    """Manage 2 Instagram accounts for cross-border e-commerce"""

    def __init__(self):
        self.client_wrapper = InstagramClientWrapper()
        self.account_repo: Optional[AccountRepository] = None
        self.active_clients: Dict[int, InstagramActions] = {}
        self.account_usage: Dict[int, int] = {}

    async def initialize(self, account_repo):
        """Initialize account manager with repository"""
        self.account_repo = account_repo

    async def setup_accounts(self, account_configs: Dict[int, Dict]):
        """Setup and initialize both accounts"""
        for account_id, config in account_configs.items():
            if not config.get('username'):
                print(f"⚠️  Account {account_id} not configured, skipping")
                continue

            try:
                client = await self.client_wrapper.initialize_account(
                    account_id=account_id,
                    username=config['username'],
                    password=config['password'],
                    proxy=config.get('proxy')
                )
                actions = InstagramActions(client)
                self.active_clients[account_id] = actions
                self.account_usage[account_id] = 0
                
                # Save to database
                await self.account_repo.create({
                    'id': account_id,
                    'username': config['username'],
                    'password_encrypted': config['password'],
                    'account_type': 'test',
                    'status': 'active',
                    'niche': 'phone_accessories',
                    'primary_category': config.get('primary_category'),
                    'secondary_categories': config.get('secondary_categories', [])
                })
                
            except Exception as e:
                print(f"❌ Failed to setup account {account_id}: {e}")

    def get_client(self, account_id: int) -> Optional[InstagramActions]:
        """Get InstagramActions client for account"""
        return self.active_clients.get(account_id)

    async def get_client_by_category(self, category: str) -> Optional[InstagramActions]:
        """Get client based on product category"""
        # Account 1: phone_case, phone_film
        # Account 2: earbuds, noise_cancelling_headphone
        # Shared: charging_cable, charger (rotate)
        
        account_1_categories = ['phone_case', 'phone_film']
        account_2_categories = ['earbuds', 'noise_cancelling_headphone']
        shared_categories = SHARED_CATEGORIES

        if category in account_1_categories:
            return self.get_client(1)
        elif category in account_2_categories:
            return self.get_client(2)
        elif category in shared_categories:
            # Rotate between accounts for shared categories
            return await self.rotate_client()
        else:
            # Default to account 1
            return self.get_client(1)

    async def rotate_client(self) -> Optional[InstagramActions]:
        """Rotate between accounts for shared categories"""
        # Select account with lower usage
        account_id = min(
            self.account_usage.items(),
            key=lambda x: x[1]
        )[0]
        
        print(f"🔄 Rotating to account {account_id}")
        return self.get_client(account_id)

    async def get_least_used_account(self) -> Optional[int]:
        """Get account with lowest usage"""
        if not self.account_usage:
            return None
        
        account_id = min(
            self.account_usage.items(),
            key=lambda x: x[1]
        )[0]
        
        return account_id

    def increment_usage(self, account_id: int):
        """Increment account usage counter"""
        if account_id in self.account_usage:
            self.account_usage[account_id] += 1
        else:
            self.account_usage[account_id] = 1

    def get_usage_stats(self) -> Dict[int, int]:
        """Get usage statistics for all accounts"""
        return self.account_usage.copy()

    async def reset_daily_usage(self):
        """Reset daily usage counters"""
        for account_id in self.account_usage:
            self.account_usage[account_id] = 0
            if self.account_repo:
                await self.account_repo.reset_daily_actions(account_id)
        print("📊 Daily usage counters reset")

    async def logout_all(self):
        """Logout all accounts"""
        for account_id in list(self.active_clients.keys()):
            await self.client_wrapper.logout_account(account_id)
        self.active_clients.clear()
        self.account_usage.clear()
        print("✅ All accounts logged out")

    def is_account_ready(self, account_id: int) -> bool:
        """Check if account is ready to use"""
        return account_id in self.active_clients
