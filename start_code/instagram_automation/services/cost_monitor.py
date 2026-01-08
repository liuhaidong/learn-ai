"""Instagram Automation - Cost Monitoring Service"""

from datetime import datetime
from typing import Dict, float
from database.connection import get_db_session
from database.repositories import APIUsageRepository


class CostMonitor:
    """Monitor and track API costs for budget management"""

    def __init__(self):
        self.monthly_budget = 100.0  # $100/month
        self.daily_budget = self.monthly_budget / 30  # $3.33/day
        self.current_month_cost = 0.0
        self.current_day_cost = 0.0
        self.today_date = datetime.now().date()

    def update_daily_budget(self, monthly_budget: float):
        """Update budget"""
        self.monthly_budget = monthly_budget
        self.daily_budget = monthly_budget / 30

    async def track_openai_usage(
        self,
        action: str,
        tokens_used: int,
        metadata: Dict = None
    ) -> None:
        """Track OpenAI usage and cost"""

        # Calculate cost: gpt-4o-mini = $0.15/1K tokens
        cost = (tokens_used / 1000) * 0.00015

        print(f"💰 OpenAI Usage:")
        print(f"   Action: {action}")
        print(f"   Tokens: {tokens_used}")
        print(f"   Cost: ${cost:.4f}")

        # Track cost
        self.current_day_cost += cost
        self.current_month_cost += cost

        # Log to database
        async with get_db_session() as session:
            api_repo = APIUsageRepository(session)
            await api_repo.create_usage_log(
                service='openai',
                action=action,
                quantity=tokens_used,
                cost_usd=cost,
                metadata=metadata or {}
            )

        self._check_budget_limits()

    async def track_midjourney_usage(
        self,
        image_count: int = 1,
        metadata: Dict = None
    ) -> None:
        """Track Midjourney usage and cost"""

        # Calculate cost: standard = $0.05, HD = $0.10 per image
        cost = image_count * 0.05  # Assume standard quality

        print(f"🎨 Midjourney Usage:")
        print(f"   Images: {image_count}")
        print(f"   Cost: ${cost:.2f}")

        # Track cost
        self.current_day_cost += cost
        self.current_month_cost += cost

        # Log to database
        async with get_db_session() as session:
            api_repo = APIUsageRepository(session)
            await api_repo.create_usage_log(
                service='midjourney',
                action='image_generation',
                quantity=image_count,
                cost_usd=cost,
                metadata=metadata or {}
            )

        self._check_budget_limits()

    async def track_total_cost(
        self,
        openai_tokens: int = 0,
        midjourney_images: int = 0,
        metadata: Dict = None
    ) -> None:
        """Track total cost for content generation"""

        openai_cost = (openai_tokens / 1000) * 0.00015
        midjourney_cost = midjourney_images * 0.05
        total_cost = openai_cost + midjourney_cost

        print(f"💵 Total Content Generation Cost:")
        print(f"   OpenAI: ${openai_cost:.4f} ({openai_tokens} tokens)")
        print(f"   Midjourney: ${midjourney_cost:.2f} ({midjourney_images} images)")
        print(f"   Total: ${total_cost:.2f}")

        # Track cost
        self.current_day_cost += total_cost
        self.current_month_cost += total_cost

        # Log to database
        async with get_db_session() as session:
            api_repo = APIUsageRepository(session)
            await api_repo.create_usage_log(
                service='openai',
                action='content_generation',
                quantity=openai_tokens,
                cost_usd=openai_cost,
                metadata=metadata or {}
            )
            await api_repo.create_usage_log(
                service='midjourney',
                action='content_generation',
                quantity=midjourney_images,
                cost_usd=midjourney_cost,
                metadata=metadata or {}
            )

        self._check_budget_limits()

    def _check_budget_limits(self):
        """Check if we're approaching or exceeding budget"""

        # Daily budget check
        if self.current_day_cost >= self.daily_budget * 0.8:
            print(f"⚠️  WARNING: 80% of daily budget used: ${self.current_day_cost:.2f}/${self.daily_budget:.2f}")

        if self.current_day_cost >= self.daily_budget:
            print(f"🚨  ALERT: Daily budget exceeded: ${self.current_day_cost:.2f}/${self.daily_budget:.2f}")

        # Monthly budget check
        if self.current_month_cost >= self.monthly_budget * 0.8:
            print(f"⚠️  WARNING: 80% of monthly budget used: ${self.current_month_cost:.2f}/${self.monthly_budget:.2f}")

        if self.current_month_cost >= self.monthly_budget:
            print(f"🚨  ALERT: Monthly budget exceeded: ${self.current_month_cost:.2f}/${self.monthly_budget:.2f}")

    async def generate_daily_report(self) -> Dict:
        """Generate daily cost report"""

        report = {
            'date': datetime.now().date(),
            'daily_cost': self.current_day_cost,
            'daily_budget': self.daily_budget,
            'budget_usage': self.current_day_cost / self.daily_budget,
            'monthly_cost': self.current_month_cost,
            'monthly_budget': self.monthly_budget,
            'budget_percentage': (self.current_month_cost / self.monthly_budget) * 100
        }

        print(f"\n" + "=" * 60)
        print(f"💰  DAILY COST REPORT - {report['date']}")
        print("=" * 60)
        print(f"Daily Cost: ${report['daily_cost']:.2f} / ${report['daily_budget']:.2f}")
        print(f"Budget Used: {report['budget_usage']*100:.1f}%")
        print(f"Monthly YTD: ${report['monthly_cost']:.2f} / ${report['monthly_budget']:.2f}")
        print(f"Monthly Used: {report['budget_percentage']:.1f}%")
        print("=" * 60)

        # Reset daily counter
        self.current_day_cost = 0.0
        self.today_date = datetime.now().date()

        return report

    async def get_cost_summary(self) -> Dict:
        """Get cost summary"""

        return {
            'monthly_budget': self.monthly_budget,
            'daily_budget': self.daily_budget,
            'current_month_cost': self.current_month_cost,
            'current_day_cost': self.current_day_cost,
            'monthly_remaining': self.monthly_budget - self.current_month_cost,
            'daily_remaining': self.daily_budget - self.current_day_cost,
            'budget_used_percentage': (self.current_month_cost / self.monthly_budget) * 100
        }
