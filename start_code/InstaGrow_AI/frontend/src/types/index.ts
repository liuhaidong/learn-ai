export interface User {
  id: string;
  email: string;
  full_name: string | null;
  created_at: string;
}

export interface Workspace {
  id: string;
  user_id: string;
  name: string;
  created_at: string;
}

export interface InstagramAccount {
  id: string;
  workspace_id: string;
  username: string;
  instagram_user_id: string | null;
  product_category: string | null;
  brand_persona: string | null;
  is_active: boolean;
  created_at: string;
}

export type ContentType = 'POST' | 'REEL' | 'STORY';
export type ContentStatus = 'DRAFT' | 'PENDING_APPROVAL' | 'SCHEDULED' | 'POSTED' | 'FAILED';

export interface ContentPiece {
  id: string;
  instagram_account_id: string;
  type: ContentType;
  status: ContentStatus;
  generated_caption: string | null;
  generated_image_url: string | null;
  suggested_hashtags: string[] | null;
  final_caption: string | null;
  final_media_url: string | null;
  scheduled_at: string | null;
  posted_at: string | null;
  post_url: string | null;
  error_message: string | null;
  ai_raw_response: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface Competitor {
  id: string;
  workspace_id: string;
  username: string;
  analysis_results: Record<string, unknown> | null;
  last_analyzed_at: string | null;
  created_at: string;
}

export type InteractionType = 'LIKE' | 'COMMENT' | 'FOLLOW';
export type InteractionStatus = 'PENDING_APPROVAL' | 'EXECUTED' | 'FAILED' | 'CANCELED';

export interface Interaction {
  id: string;
  instagram_account_id: string;
  target_username: string;
  target_post_url: string | null;
  type: InteractionType;
  status: InteractionStatus;
  content: string | null;
  ai_generated_comment: string | null;
  executed_at: string | null;
  error_message: string | null;
  created_at: string;
}

export interface AnalyticsSnapshot {
  id: string;
  instagram_account_id: string;
  snapshot_date: string;
  followers_count: number | null;
  following_count: number | null;
  posts_count: number | null;
  profile_views: number | null;
  website_clicks: number | null;
  created_at: string;
}

export interface CompetitorAnalysis {
  bio: string;
  follower_count: number;
  following_count: number;
  posts_count: number;
  engagement_rate: number;
  content_strategy: {
    reels_percentage: number;
    carousel_percentage: number;
    single_image_percentage: number;
    story_percentage: number;
  };
  top_hashtags: string[];
  best_posting_times: string[];
  value_proposition: string;
}
