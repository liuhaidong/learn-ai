"""Instagram Automation - Database Repositories (Fixed Version)"""

from typing import List, Optional, Dict, Any
from sqlalchemy import select, and_, func, desc, update
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import (
    Account, ProductCategory, ProductPositioning,
    CompetitorAccount, TargetUser, ContentTask,
    Interaction, Location, SystemLog, TaskQueue, APIUsageLog
)
from datetime import datetime, timedelta
from sqlalchemy.dialects.postgresql import insert


class BaseRepository:
    """Base repository with common methods"""

    def __init__(self, session: AsyncSession):
        self.session = session


class AccountRepository(BaseRepository):
    """Repository for Account model"""

    async def get_by_id(self, account_id: int) -> Optional[Account]:
        """Get account by ID"""
        result = await self.session.execute(
            select(Account).where(Account.id == account_id)
        )
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> Optional[Account]:
        """Get account by username"""
        result = await self.session.execute(
            select(Account).where(Account.username == username)
        )
        return result.scalar_one_or_none()

    async def get_all_active(self, account_type: str = None) -> List[Account]:
        """Get all active accounts, optionally filtered by type"""
        query = select(Account).where(Account.status == "active")
        if account_type:
            query = query.where(Account.account_type == account_type)
        result = await self.session.execute(query.order_by(Account.usage_count))
        return result.scalars().all()

    async def get_account_by_category(
        self,
        category: str,
        account_type: str = "test"
    ) -> Optional[Account]:
        """Get account by product category"""
        query = select(Account).where(
            and_(
                Account.account_type == account_type,
                Account.status == "active",
                (Account.primary_category == category) |
                (Account.secondary_categories.isnot(None))
            )
        )
        return (await self.session.execute(query)).scalar_one_or_none()

    async def create(self, account_data: Dict[str, Any]) -> Account:
        """Create new account"""
        account = Account(**account_data)
        self.session.add(account)
        await self.session.flush()
        return account

    async def update(self, account_id: int, update_data: Dict[str, Any]) -> Account:
        """Update account"""
        result = await self.session.execute(
            select(Account).where(Account.id == account_id)
        )
        account = result.scalar_one()
        for key, value in update_data.items():
            setattr(account, key, value)
        account.updated_at = datetime.utcnow()
        await self.session.flush()
        return account

    async def increment_usage(self, account_id: int) -> None:
        """Increment account usage count"""
        result = await self.session.execute(
            select(Account).where(Account.id == account_id)
        )
        account = result.scalar_one()
        account.usage_count += 1
        account.daily_actions += 1
        await self.session.flush()

    async def reset_daily_actions(self, account_id: int) -> None:
        """Reset daily action count (call at midnight)"""
        result = await self.session.execute(
            update(Account).where(Account.id == account_id).values(daily_actions=0)
        )
        await self.session.flush()

    async def get_least_used_account(
        self,
        category: str = None
    ) -> Optional[Account]:
        """Get account with lowest usage count"""
        query = select(Account).where(Account.status == "active")
        if category:
            query = query.where(
                (Account.primary_category == category) |
                (Account.secondary_categories.isnot(None))
            )
        result = await self.session.execute(query.order_by(Account.daily_actions.asc()).limit(1))
        return result.scalar_one_or_none()


class ProductCategoryRepository(BaseRepository):
    """Repository for ProductCategory model"""

    async def get_by_name(self, category_name: str) -> Optional[ProductCategory]:
        """Get product category by name"""
        result = await self.session.execute(
            select(ProductCategory).where(
                ProductCategory.category_name == category_name
            )
        )
        return result.scalar_one_or_none()

    async def get_all(self) -> List[ProductCategory]:
        """Get all product categories"""
        result = await self.session.execute(
            select(ProductCategory).order_by(ProductCategory.id)
        )
        return result.scalars().all()

    async def create_or_update(
        self,
        category_name: str,
        category_data: Dict[str, Any]
    ) -> ProductCategory:
        """Create or update product category"""
        result = await self.session.execute(
            select(ProductCategory).where(
                ProductCategory.category_name == category_name
            )
        )
        category = result.scalar_one_or_none()

        if category:
            # Update existing
            for key, value in category_data.items():
                setattr(category, key, value)
            category.updated_at = datetime.utcnow()
        else:
            # Create new
            category = ProductCategory(
                category_name=category_name,
                **category_data
            )
            self.session.add(category)

        await self.session.flush()
        return category


class CompetitorRepository(BaseRepository):
    """Repository for CompetitorAccount model"""

    async def get_active_competitors(
        self,
        account_id: int,
        category: str = None,
        limit: int = 20
    ) -> List[CompetitorAccount]:
        """Get active competitors for an account"""
        query = select(CompetitorAccount).where(
            CompetitorAccount.account_id == account_id
        )
        if category:
            query = query.where(CompetitorAccount.category == category)
        result = await self.session.execute(
            query.order_by(desc(CompetitorAccount.engagement_score)).limit(limit)
        )
        return result.scalars().all()

    async def create_or_update(
        self,
        competitor_data: Dict[str, Any]
    ) -> CompetitorAccount:
        """Create or update competitor"""
        result = await self.session.execute(
            select(CompetitorAccount).where(
                and_(
                    CompetitorAccount.account_id == competitor_data['account_id'],
                    CompetitorAccount.competitor_username == competitor_data['competitor_username']
                )
            )
        )
        competitor = result.scalar_one_or_none()

        if competitor:
            # Update existing
            for key, value in competitor_data.items():
                if key not in ['account_id', 'competitor_username']:
                    setattr(competitor, key, value)
            competitor.updated_at = datetime.utcnow()
        else:
            # Create new
            competitor = CompetitorAccount(**competitor_data)
            self.session.add(competitor)

        await self.session.flush()
        return competitor

    async def get_competitors_to_analyze(
        self,
        account_id: int,
        hours: int = 24
    ) -> List[CompetitorAccount]:
        """Get competitors that need analysis"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(CompetitorAccount).where(
                and_(
                    CompetitorAccount.account_id == account_id,
                    (CompetitorAccount.last_analyzed_at.is_(None)) |
                    (CompetitorAccount.last_analyzed_at < cutoff_time)
                )
            )
        )
        return result.scalars().all()


class TargetUserRepository(BaseRepository):
    """Repository for TargetUser model"""

    async def get_pending_users(
        self,
        account_id: int,
        limit: int = 100,
        category: str = None
    ) -> List[TargetUser]:
        """Get pending target users for interaction"""
        query = select(TargetUser).where(
            and_(
                TargetUser.account_id == account_id,
                TargetUser.status == "pending"
            )
        )
        if category:
            query = query.where(TargetUser.category == category)
        result = await self.session.execute(
            query.order_by(desc(TargetUser.engagement_score)).limit(limit)
        )
        return result.scalars().all()

    async def get_users_to_interact(
        self,
        account_id: int,
        limit: int = 50
    ) -> List[TargetUser]:
        """Get users that need interaction today"""
        today = datetime.utcnow().date()
        result = await self.session.execute(
            select(TargetUser).where(
                and_(
                    TargetUser.account_id == account_id,
                    TargetUser.status.in_(["pending", "interacted"]),
                    (
                        TargetUser.last_interaction_at.is_(None) |
                        (func.date(TargetUser.last_interaction_at) < today)
                    ),
                    TargetUser.daily_interaction_count < 2  # Max 2 interactions per user per day
                )
            )
        )
        return result.scalars().all()

    async def create_or_update(
        self,
        user_data: Dict[str, Any]
    ) -> TargetUser:
        """Create or update target user"""
        result = await self.session.execute(
            select(TargetUser).where(
                and_(
                    TargetUser.account_id == user_data['account_id'],
                    TargetUser.user_id == user_data.get('user_id')
                )
            )
        )
        user = result.scalar_one_or_none()

        if user:
            # Update existing
            for key, value in user_data.items():
                if key not in ['account_id', 'user_id']:
                    setattr(user, key, value)
            user.updated_at = datetime.utcnow()
        else:
            # Create new
            user = TargetUser(**user_data)
            self.session.add(user)

        await self.session.flush()
        return user

    async def update_interaction_count(
        self,
        user_id: int,
        status: str = "interacted"
    ) -> None:
        """Update user interaction count and status"""
        result = await self.session.execute(
            select(TargetUser).where(TargetUser.id == user_id)
        )
        user = result.scalar_one()
        if user:
            user.interaction_count += 1
            user.daily_interaction_count += 1
            user.last_interaction_at = datetime.utcnow()
            user.status = status
            user.updated_at = datetime.utcnow()
            await self.session.flush()

    async def get_today_count(
        self,
        account_id: int,
        user_id: int
    ) -> int:
        """Get today's interaction count for a user"""
        today = datetime.utcnow().date()
        result = await self.session.execute(
            select(func.count(TargetUser.id)).where(
                and_(
                    TargetUser.account_id == account_id,
                    TargetUser.user_id == user_id,
                    func.date(TargetUser.last_interaction_at) == today
                )
            )
        )
        return result.scalar() or 0

    async def reset_daily_counts(self, account_id: int) -> None:
        """Reset daily interaction counts for all users of an account"""
        await self.session.execute(
            update(TargetUser).where(TargetUser.account_id == account_id).values(daily_interaction_count=0)
        )
        await self.session.commit()


class ContentTaskRepository(BaseRepository):
    """Repository for ContentTask model"""

    async def get_ready_tasks(
        self,
        account_id: int,
        limit: int = 10
    ) -> List[ContentTask]:
        """Get tasks ready to post"""
        now = datetime.utcnow()
        result = await self.session.execute(
            select(ContentTask).where(
                and_(
                    ContentTask.account_id == account_id,
                    ContentTask.status == "ready",
                    (ContentTask.scheduled_at <= now)
                )
            )
        )
        return result.scalars().all()

    async def get_pending_tasks(
        self,
        account_id: int,
        limit: int = 5
    ) -> List[ContentTask]:
        """Get pending content tasks"""
        result = await self.session.execute(
            select(ContentTask).where(
                and_(
                    ContentTask.account_id == account_id,
                    ContentTask.status == "pending"
                )
            )
        )
        return result.scalars().all()

    async def create_task(
        self,
        task_data: Dict[str, Any]
    ) -> ContentTask:
        """Create new content task"""
        task = ContentTask(**task_data)
        task.created_at = datetime.utcnow()
        self.session.add(task)
        await self.session.flush()
        return task

    async def update_task_status(
        self,
        task_id: int,
        status: str,
        **kwargs
    ) -> ContentTask:
        """Update task status"""
        result = await self.session.execute(
            select(ContentTask).where(ContentTask.id == task_id)
        )
        task = result.scalar_one()
        task.status = status
        task.updated_at = datetime.utcnow()
        for key, value in kwargs.items():
            setattr(task, key, value)
        await self.session.flush()
        return task


class InteractionRepository(BaseRepository):
    """Repository for Interaction model"""

    async def create(
        self,
        interaction_data: Dict[str, Any]
    ) -> Interaction:
        """Create new interaction log"""
        interaction = Interaction(**interaction_data)
        interaction.created_at = datetime.utcnow()
        self.session.add(interaction)
        await self.session.flush()
        return interaction

    async def get_interactions_today(
        self,
        account_id: int,
        interaction_type: str = None
    ) -> int:
        """Get today's interaction count"""
        today = datetime.utcnow().date()
        query = select(func.count(Interaction.id)).where(
            and_(
                Interaction.account_id == account_id,
                func.date(Interaction.created_at) == today
            )
        )
        if interaction_type:
            query = query.where(Interaction.interaction_type == interaction_type)
        result = await self.session.execute(query)
        return result.scalar() or 0


class APIUsageRepository(BaseRepository):
    """Repository for API usage and cost tracking"""

    async def create_usage_log(
        self,
        service: str,
        action: str,
        quantity: int,
        cost_usd: float,
        metadata: Dict[str, Any] = None
    ) -> APIUsageLog:
        """Create API usage log"""
        log = APIUsageLog(
            service=service,
            action=action,
            quantity=quantity,
            cost_usd=cost_usd,
            metadata=metadata or {}
        )
        log.created_at = datetime.utcnow()
        self.session.add(log)
        await self.session.flush()
        return log

    async def get_daily_cost(
        self,
        service: str = None
    ) -> float:
        """Get today's total cost"""
        today = datetime.utcnow().date()
        query = select(func.sum(APIUsageLog.cost_usd)).where(
            func.date(APIUsageLog.created_at) == today
        )
        if service:
            query = query.where(APIUsageLog.service == service)
        result = await self.session.execute(query)
        return result.scalar() or 0.0

    async def get_monthly_cost(
        self,
        service: str = None
    ) -> float:
        """Get this month's total cost"""
        today = datetime.utcnow()
        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        query = select(func.sum(APIUsageLog.cost_usd)).where(
            APIUsageLog.created_at >= month_start
        )
        if service:
            query = query.where(APIUsageLog.service == service)
        result = await self.session.execute(query)
        return result.scalar() or 0.0


class SystemLogRepository(BaseRepository):
    """Repository for system logging"""

    async def create_log(
        self,
        account_id: Optional[int],
        log_level: str,
        module: str,
        message: str,
        details: Dict[str, Any] = None
    ) -> SystemLog:
        """Create system log"""
        log = SystemLog(
            account_id=account_id,
            log_level=log_level,
            module=module,
            message=message,
            details=details or {}
        )
        log.created_at = datetime.utcnow()
        self.session.add(log)
        await self.session.flush()
        return log
