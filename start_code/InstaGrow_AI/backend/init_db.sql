-- Database initialization script for InstaGrow AI
-- PostgreSQL Schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Define ENUM types
DO $$ BEGIN
    CREATE TYPE content_type AS ENUM ('POST', 'REEL', 'STORY');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE content_status AS ENUM ('DRAFT', 'PENDING_APPROVAL', 'SCHEDULED', 'POSTED', 'FAILED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE interaction_type AS ENUM ('LIKE', 'COMMENT', 'FOLLOW');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE interaction_status AS ENUM ('PENDING_APPROVAL', 'EXECUTED', 'FAILED', 'CANCELED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 1. Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Workspaces table
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. Instagram accounts table
CREATE TABLE IF NOT EXISTS instagram_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    username VARCHAR(100) UNIQUE NOT NULL,
    instagram_user_id VARCHAR(100),
    product_category TEXT,
    brand_persona TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4. Competitors table
CREATE TABLE IF NOT EXISTS competitors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    username VARCHAR(100) NOT NULL,
    analysis_results JSONB,
    last_analyzed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, username)
);

-- 5. Content pieces table
CREATE TABLE IF NOT EXISTS content_pieces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instagram_account_id UUID NOT NULL REFERENCES instagram_accounts(id) ON DELETE CASCADE,
    type content_type NOT NULL,
    status content_status NOT NULL DEFAULT 'DRAFT',
    
    -- AI-generated content
    generated_caption TEXT,
    generated_image_url TEXT,
    suggested_hashtags TEXT[],
    
    -- Final confirmed content
    final_caption TEXT,
    final_media_url TEXT,
    
    -- Scheduling and publishing
    scheduled_at TIMESTAMPTZ,
    posted_at TIMESTAMPTZ,
    post_url VARCHAR(255),
    error_message TEXT,
    
    -- Debug and tracking
    ai_raw_response JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 6. Interactions table
CREATE TABLE IF NOT EXISTS interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instagram_account_id UUID NOT NULL REFERENCES instagram_accounts(id) ON DELETE CASCADE,
    target_username VARCHAR(100) NOT NULL,
    target_post_url VARCHAR(255),
    
    type interaction_type NOT NULL,
    status interaction_status NOT NULL DEFAULT 'PENDING_APPROVAL',
    
    content TEXT,
    ai_generated_comment TEXT,
    
    executed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 7. Analytics snapshots table
CREATE TABLE IF NOT EXISTS analytics_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instagram_account_id UUID NOT NULL REFERENCES instagram_accounts(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    followers_count INT,
    following_count INT,
    posts_count INT,
    profile_views INT,
    website_clicks INT,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (instagram_account_id, snapshot_date)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_workspaces_user_id ON workspaces(user_id);
CREATE INDEX IF NOT EXISTS idx_ig_accounts_workspace_id ON instagram_accounts(workspace_id);
CREATE INDEX IF NOT EXISTS idx_ig_accounts_username ON instagram_accounts(username);
CREATE INDEX IF NOT EXISTS idx_competitors_workspace_id ON competitors(workspace_id);
CREATE INDEX IF NOT EXISTS idx_content_pieces_account_id_status ON content_pieces(instagram_account_id, status);
CREATE INDEX IF NOT EXISTS idx_content_pieces_account_id_scheduled ON content_pieces(instagram_account_id, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_interactions_account_id_status ON interactions(instagram_account_id, status);
CREATE INDEX IF NOT EXISTS idx_interactions_target_username ON interactions(target_username);
CREATE INDEX IF NOT EXISTS idx_analytics_account_id_date ON analytics_snapshots(instagram_account_id, snapshot_date DESC);

-- Create trigger function for updating updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to content_pieces table
DROP TRIGGER IF EXISTS update_content_pieces_updated_at ON content_pieces;
CREATE TRIGGER update_content_pieces_updated_at
    BEFORE UPDATE ON content_pieces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
