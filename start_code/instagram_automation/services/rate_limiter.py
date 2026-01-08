"""Instagram Automation - Rate Limiter for Multi-Account"""

import asyncio
from datetime import datetime, timedelta
from typing import Tuple, Optional
from config.settings import settings


class TwoAccountRateLimiter:
    """Rate limiter for 2 Instagram accounts"""

    def __init__(self):
        # Per-account limits
        self.account_limits = {
            'likes_per_hour': settings.max_likes_per_hour,
            'follows_per_hour': settings.max_follows_per_hour,
            'comments_per_hour': settings.max_comments_per_hour,
            'posts_per_day': settings.max_posts_per_day,
            'daily_interactions': settings.max_daily_interactions
        }

        # Global limits (2 accounts combined)
        self.global_limits = {
            'total_likes_per_hour': settings.max_likes_per_hour * 2,
            'total_follows_per_hour': settings.max_follows_per_hour * 2,
            'total_interactions_per_day': settings.max_daily_interactions * 2
        }

        # Tracking
        self.account_actions: dict = {}  # account_id -> {action -> count}
        self.hourly_reset = datetime.utcnow()
        self.daily_reset = datetime.utcnow()

    async def can_perform_action(
        self,
        account_id: int,
        action: str
    ) -> Tuple[bool, int]:
        """Check if action can be performed, returns (can_perform, wait_seconds)"""

        # Reset counters if needed
        await self._reset_counters_if_needed()

        # Check account-level limits
        limit_key = f"{action}_per_hour"
        account_limit = self.account_limits.get(limit_key, 10)

        if account_id not in self.account_actions:
            self.account_actions[account_id] = {}

        current_count = self.account_actions[account_id].get(action, 0)

        if current_count >= account_limit:
            # Account exceeded hourly limit
            wait_time = self._get_time_until_next_hour()
            print(f"⚠️ Account {account_id} exceeded {action} limit")
            return (False, wait_time)

        # Check global limits
        global_count = await self._get_global_action_count(action)
        global_limit = self.global_limits.get(f"total_{action}_per_hour", account_limit * 2)

        if global_count >= global_limit:
            # Global limit exceeded
            wait_time = self._get_time_until_next_hour()
            print(f"⚠️ Global {action} limit exceeded")
            return (False, wait_time)

        # Check daily interaction limit
        if action in ['like', 'comment', 'follow']:
            daily_count = await self._get_daily_interaction_count(account_id)
            if daily_count >= self.account_limits['daily_interactions']:
                wait_time = self._get_time_until_midnight()
                print(f"⚠️ Account {account_id} daily interaction limit exceeded")
                return (False, wait_time)

        # All checks passed
        print(f"✅ Action {action} allowed for account {account_id}")
        return (True, 0)

    async def record_action(self, account_id: int, action: str):
        """Record that an action was performed"""

        if account_id not in self.account_actions:
            self.account_actions[account_id] = {}

        self.account_actions[account_id][action] = \
            self.account_actions[account_id].get(action, 0) + 1

        print(f"📊 Recorded {action} for account {account_id}")

    async def _get_global_action_count(self, action: str) -> int:
        """Get total action count across all accounts for current hour"""
        limit_key = f"{action}_per_hour"
        total = 0

        for account_actions in self.account_actions.values():
            total += account_actions.get(action, 0)

        return total

    async def _get_daily_interaction_count(self, account_id: int) -> int:
        """Get daily interaction count for an account"""
        if account_id not in self.account_actions:
            return 0

        total = sum(
            self.account_actions[account_id].get(action, 0)
            for action in ['like', 'comment', 'follow']
        )

        return total

    async def _reset_counters_if_needed(self):
        """Reset hourly and daily counters if time has passed"""

        now = datetime.utcnow()

        # Reset hourly
        if (now - self.hourly_reset).total_seconds() >= 3600:
            self.account_actions.clear()
            self.hourly_reset = now
            print("🕐 Hourly counters reset")

        # Reset daily
        if (now - self.daily_reset).total_seconds() >= 86400:
            self.daily_reset = now
            print("🌅 Daily counters reset")

    def _get_time_until_next_hour(self) -> int:
        """Get seconds until next hour"""
        now = datetime.utcnow()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return int((next_hour - now).total_seconds())

    def _get_time_until_midnight(self) -> int:
        """Get seconds until midnight"""
        now = datetime.utcnow()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return int((midnight - now).total_seconds())

    def get_status(self) -> dict:
        """Get current status of all accounts"""

        now = datetime.utcnow()

        status = {
            'accounts': {},
            'hourly_reset_in': self._get_time_until_next_hour(),
            'daily_reset_in': self._get_time_until_midnight()
        }

        for account_id, actions in self.account_actions.items():
            status['accounts'][account_id] = {
                'hourly_actions': sum(actions.values()),
                'by_action': actions.copy()
            }

        return status

    async def wait_if_needed(self, account_id: int, action: str):
        """Wait if rate limit requires it"""

        can_perform, wait_time = await self.can_perform_action(account_id, action)

        if not can_perform and wait_time > 0:
            print(f"⏸️  Rate limited, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
