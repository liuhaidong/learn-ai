"""Instagram Automation - Database Models"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Text, Float, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.orm import relationship

Base = declarative_base()


class Account(Base):
    """Instagram account model for multi-account management"""
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    password_encrypted = Column(Text, nullable=False)
    session_data = Column(JSON, nullable=True)
    proxy_config = Column(JSON, nullable=True)
    status = Column(String(50), default="active")  # active, banned, warming, inactive
    account_type = Column(String(50), default="test")  # test, production, staging
    niche = Column(String(100), default="phone_accessories")
    primary_category = Column(String(100), nullable=True)
    secondary_categories = Column(JSON, nullable=True)
    settings = Column(JSON, nullable=True)
    usage_count = Column(Integer, default=0)
    daily_actions = Column(Integer, default=0)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    content_tasks = relationship("ContentTask", back_populates="account", cascade="all, delete-orphan")
    competitor_accounts = relationship("CompetitorAccount", back_populates="account", cascade="all, delete-orphan")
    target_users = relationship("TargetUser", back_populates="account", cascade="all, delete-orphan")
    interactions = relationship("Interaction", back_populates="account", cascade="all, delete-orphan")
    product_positioning = relationship("ProductPositioning", back_populates="account", uselist=False)


class ProductCategory(Base):
    """Product category configuration for 6 categories"""
    __tablename__ = "product_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String(100), unique=True, nullable=False, index=True)  # charging_cable, charger, etc.
    display_name = Column(String(200), nullable=False)
    content_themes = Column(JSON, nullable=True)
    hashtags = Column(JSON, nullable=True)
    price_range = Column(JSON, nullable=True)
    key_features = Column(JSON, nullable=True)
    target_audience = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProductPositioning(Base):
    """Product positioning configuration for each account"""
    __tablename__ = "product_positioning"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False, unique=True)
    niche = Column(String(100), nullable=False)
    target_audience = Column(JSON, nullable=True)
    content_themes = Column(JSON, nullable=True)
    hashtags = Column(JSON, nullable=True)
    tone_of_voice = Column(String(100), default="professional")
    language = Column(String(10), default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    account = relationship("Account", back_populates="product_positioning")


class CompetitorAccount(Base):
    """Competitor account tracking"""
    __tablename__ = "competitor_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    competitor_username = Column(String(255), nullable=False, index=True)
    competitor_id = Column(Integer, nullable=True)
    follow_status = Column(Boolean, default=False)
    analysis_data = Column(JSON, nullable=True)
    category = Column(String(100), nullable=True)  # Which product category
    last_analyzed_at = Column(DateTime, nullable=True)
    engagement_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    account = relationship("Account", back_populates="competitor_accounts")

    # Indexes
    __table_args__ = (
        Index("idx_competitor_account", "account_id", "category"),
    )


class TargetUser(Base):
    """Target user for interaction"""
    __tablename__ = "target_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    username = Column(String(255), nullable=False)
    user_id = Column(Integer, nullable=True)
    engagement_score = Column(Integer, default=0)
    last_interaction_at = Column(DateTime, nullable=True)
    interaction_count = Column(Integer, default=0)
    daily_interaction_count = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, interacted, followed, blocked
    category = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    account = relationship("Account", back_populates="target_users")

    # Indexes
    __table_args__ = (
        Index("idx_target_user_account", "account_id", "status"),
        Index("idx_target_user_category", "account_id", "category"),
    )


class ContentTask(Base):
    """Content generation and posting task"""
    __tablename__ = "content_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("product_categories.id"), nullable=True)
    task_type = Column(String(50), nullable=False)  # product_review, value_deal, etc.
    status = Column(String(50), default="pending")  # pending, generating, ready, posted, failed
    generated_caption = Column(Text, nullable=True)
    generated_image_url = Column(Text, nullable=True)
    hashtags = Column(JSON, nullable=True)
    media_type = Column(String(50), default="photo")
    scheduled_at = Column(DateTime, nullable=True)
    posted_at = Column(DateTime, nullable=True)
    posted_media_id = Column(String(255), nullable=True)
    product_info = Column(JSON, nullable=True)  # Product details for e-commerce
    error_message = Column(Text, nullable=True)
    cost_tracking = Column(JSON, nullable=True)  # API cost tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    account = relationship("Account", back_populates="content_tasks")

    # Indexes
    __table_args__ = (
        Index("idx_content_task_account", "account_id", "status"),
        Index("idx_content_task_scheduled", "scheduled_at"),
    )


class Interaction(Base):
    """Interaction log for tracking engagement"""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    target_user_id = Column(Integer, ForeignKey("target_users.id"), nullable=True)
    interaction_type = Column(String(50), nullable=False)  # like, comment, follow, dm
    media_id = Column(String(255), nullable=True)
    comment_content = Column(Text, nullable=True)
    status = Column(String(50), default="success")  # success, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    account = relationship("Account", back_populates="interactions")

    # Indexes
    __table_args__ = (
        Index("idx_interaction_account", "account_id", "interaction_type"),
        Index("idx_interaction_target", "target_user_id"),
    )


class Location(Base):
    """Location information for hashtag and location-based targeting"""
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(String(255), nullable=True)
    location_name = Column(String(255), nullable=False)
    address = Column(Text, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    analysis_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLog(Base):
    """System log for debugging and monitoring"""
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    log_level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    module = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("idx_log_level", "log_level"),
        Index("idx_log_module", "module"),
    )


class TaskQueue(Base):
    """Task queue for job management"""
    __tablename__ = "task_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    task_type = Column(String(100), nullable=False)
    task_data = Column(JSON, nullable=False)
    priority = Column(Integer, default=5)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    scheduled_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("idx_task_queue_status", "status", "scheduled_at"),
        Index("idx_task_queue_priority", "priority"),
    )


class APIUsageLog(Base):
    """API usage and cost tracking"""
    __tablename__ = "api_usage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    service = Column(String(50), nullable=False)  # openai, midjourney
    action = Column(String(100), nullable=False)  # caption, image, comment
    quantity = Column(Integer, default=0)  # tokens, images, etc.
    cost_usd = Column(Float, default=0.0)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("idx_api_usage_service", "service", "created_at"),
    )
