"""Instagram Automation - Interaction Service"""

import random
import asyncio
from typing import Dict, List
from instagram.account_manager import TwoAccountManager
from instagram.client import InstagramActions
from services.rate_limiter import TwoAccountRateLimiter
from ai.openai_client import OpenAIClient
from config.constants import INTERACTION_TYPES, INTERACTION_SCHEDULE


class InteractionService:
    """Automated interaction service for 2 accounts"""

    def __init__(self, account_manager: TwoAccountManager, rate_limiter: TwoAccountRateLimiter):
        self.account_manager = account_manager
        self.rate_limiter = rate_limiter
        self.openai_client = OpenAIClient()

    async def daily_interaction_task(self, account_id: int):
        """Execute daily interaction task for an account"""

        print(f"\n👥 Starting daily interactions for account {account_id}")
        print("=" * 60)

        # Get client
        client = self.account_manager.get_client(account_id)
        if not client:
            print(f"❌ No active client found for account {account_id}")
            return

        # Determine interaction counts based on schedule
        total_interactions = random.randint(30, 50)

        # Distribute interactions by time period
        morning_count = int(total_interactions * INTERACTION_SCHEDULE['morning']['percentage'])
        afternoon_count = int(total_interactions * INTERACTION_SCHEDULE['afternoon']['percentage'])
        evening_count = total_interactions - morning_count - afternoon_count

        print(f"📊 Today's plan:")
        print(f"   Morning: {morning_count} interactions")
        print(f"   Afternoon: {afternoon_count} interactions")
        print(f"   Evening: {evening_count} interactions")
        print(f"   Total: {total_interactions} interactions")
        print("=" * 60)

        # Execute interactions
        total_completed = 0

        # Morning interactions
        total_completed += await self._execute_interaction_batch(
            client=client,
            account_id=account_id,
            count=morning_count,
            period='morning'
        )

        await asyncio.sleep(random.randint(1800, 3600))  # 30-60 min break

        # Afternoon interactions
        total_completed += await self._execute_interaction_batch(
            client=client,
            account_id=account_id,
            count=afternoon_count,
            period='afternoon'
        )

        await asyncio.sleep(random.randint(1800, 3600))  # 30-60 min break

        # Evening interactions
        total_completed += await self._execute_interaction_batch(
            client=client,
            account_id=account_id,
            count=evening_count,
            period='evening'
        )

        print(f"\n✅ Daily interactions complete: {total_completed}/{total_interactions}")

    async def _execute_interaction_batch(
        self,
        client: InstagramActions,
        account_id: int,
        count: int,
        period: str
    ) -> int:
        """Execute a batch of interactions"""

        print(f"\n⏰ Executing {period} batch: {count} interactions")

        completed = 0
        failed = 0

        for i in range(count):
            # Check rate limit
            can_perform, wait_time = await self.rate_limiter.can_perform_action(account_id, 'like')

            if not can_perform:
                print(f"⏸️  Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)

            # Select interaction type
            interaction_type = random.choices(
                list(INTERACTION_TYPES.keys()),
                weights=list(INTERACTION_TYPES.values()),
                k=1
            )[0]

            # Execute interaction
            success = await self._perform_interaction(
                client=client,
                account_id=account_id,
                interaction_type=interaction_type
            )

            if success:
                await self.rate_limiter.record_action(account_id, interaction_type)
                completed += 1
            else:
                failed += 1

            # Random delay between actions (5-30 seconds)
            await asyncio.sleep(random.uniform(5, 30))

        print(f"   ✅ Completed: {completed}")
        print(f"   ❌ Failed: {failed}")

        return completed

    async def _perform_interaction(
        self,
        client: InstagramActions,
        account_id: int,
        interaction_type: str
    ) -> bool:
        """Perform a single interaction"""

        try:
            if interaction_type == 'like':
                # Need to find a post to like
                # For now, return False (would need to fetch feed)
                print(f"   ❤️  Liking post...")
                return False

            elif interaction_type == 'comment':
                comment = await self._generate_comment_for_category(account_id)
                if comment:
                    print(f"   💬  Commenting with: {comment[:50]}...")
                    # For now, return False (would need media_id)
                    return False

            elif interaction_type == 'follow':
                # For now, return False (would need target user)
                print(f"   👤  Following user...")
                return False

            return True

        except Exception as e:
            print(f"   ❌  Interaction failed: {e}")
            return False

    async def _generate_comment_for_category(self, account_id: int) -> str:
        """Generate category-specific comment using AI"""

        # Determine category based on account
        # Account 1: phone_case, phone_film
        # Account 2: earbuds, noise_cancelling_headphone

        if account_id == 1:
            categories = ['phone_case', 'phone_film']
        else:
            categories = ['earbuds', 'noise_cancelling_headphone']

        category = random.choice(categories)

        result = await self.openai_client.generate_comment_ecommerce(
            category=category,
            context="interested in this product, thinking of buying"
        )

        return result['comment']
