from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from app.models.models import ContentType, ContentStatus, InteractionType, InteractionStatus


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class User(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


class WorkspaceBase(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class WorkspaceCreate(WorkspaceBase):
    pass


class Workspace(WorkspaceBase):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class InstagramAccountBase(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    product_category: Optional[str] = None
    brand_persona: Optional[str] = None


class InstagramAccountCreate(InstagramAccountBase):
    workspace_id: str


class InstagramAccountUpdate(BaseModel):
    product_category: Optional[str] = None
    brand_persona: Optional[str] = None
    is_active: Optional[bool] = None


class InstagramAccount(InstagramAccountBase):
    id: str
    workspace_id: str
    instagram_user_id: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class CompetitorBase(BaseModel):
    username: str = Field(min_length=1, max_length=100)


class CompetitorCreate(CompetitorBase):
    workspace_id: str


class Competitor(CompetitorBase):
    id: str
    workspace_id: str
    analysis_results: Optional[dict] = None
    last_analyzed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CompetitorAnalysisRequest(BaseModel):
    usernames: List[str] = Field(min_length=1, max_length=20)


class ContentPieceBase(BaseModel):
    type: ContentType
    generated_caption: Optional[str] = None
    generated_image_url: Optional[str] = None
    suggested_hashtags: Optional[List[str]] = None


class ContentPieceCreate(ContentPieceBase):
    instagram_account_id: str


class ContentPieceUpdate(BaseModel):
    final_caption: Optional[str] = None
    final_media_url: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    status: Optional[ContentStatus] = None


class ContentPieceGenerateRequest(BaseModel):
    instagram_account_id: str
    type: ContentType
    product_description: str
    brand_tone: Optional[str] = "professional"
    target_audience: Optional[str] = None


class ContentPiece(ContentPieceBase):
    id: str
    instagram_account_id: str
    status: ContentStatus
    final_caption: Optional[str] = None
    final_media_url: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    posted_at: Optional[datetime] = None
    post_url: Optional[str] = None
    error_message: Optional[str] = None
    ai_raw_response: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class InteractionBase(BaseModel):
    target_username: str = Field(min_length=1, max_length=100)
    type: InteractionType


class InteractionCreate(InteractionBase):
    instagram_account_id: str
    target_post_url: Optional[str] = None
    content: Optional[str] = None


class InteractionUpdate(BaseModel):
    status: Optional[InteractionStatus] = None


class InteractionApproveRequest(BaseModel):
    interaction_ids: List[str] = Field(min_length=1)


class Interaction(InteractionBase):
    id: str
    instagram_account_id: str
    target_post_url: Optional[str] = None
    status: InteractionStatus
    content: Optional[str] = None
    ai_generated_comment: Optional[str] = None
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AnalyticsSnapshotBase(BaseModel):
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    posts_count: Optional[int] = None
    profile_views: Optional[int] = None
    website_clicks: Optional[int] = None


class AnalyticsSnapshotCreate(AnalyticsSnapshotBase):
    instagram_account_id: str
    snapshot_date: datetime


class AnalyticsSnapshot(AnalyticsSnapshotBase):
    id: str
    instagram_account_id: str
    snapshot_date: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class DiscoveryRequest(BaseModel):
    instagram_account_id: str
    limit: int = Field(default=50, ge=1, le=100)
