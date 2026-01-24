# Instagram账号运营内容模块化中台数据库设计

针对上述架构，为每个子模块提供完整的数据库设计方案，采用**PostgreSQL**作为主数据库，**Qdrant**作为向量数据库，**Redis**作为缓存层。

---

## **一、钩子工厂子模块体系数据库设计**

### **1.1 钩子库管理器**

#### **业务对象**
- **钩子(Hook)**: 可复用的开场文案单元
- **钩子标签(HookTag)**: 钩子的分类维度
- **竞品监控源(CompetitorSource)**: 爬取的竞品账号/视频

#### **核心表结构**

```sql
-- 钩子主表
CREATE TABLE hooks (
    hook_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,  -- 商家隔离
    content TEXT NOT NULL,  -- 钩子文案（如"Stop doing this!"）
    platform VARCHAR(20) CHECK (platform IN ('instagram', 'tiktok', 'youtube')), 
    hook_type VARCHAR(20) CHECK (hook_type IN ('question', 'threat', 'fact', 'curiosity', 'urgency')),
    emotion_tag VARCHAR(20),  -- surprise/curiosity/fear
    length_category VARCHAR(10) CHECK (length_category IN ('short', 'mid', 'long')),
    language VARCHAR(10) DEFAULT 'en',
    
    -- 效果数据（每日聚合更新）
    avg_ctr DECIMAL(5,4),
    avg_completion_rate DECIMAL(5,4),
    usage_count INTEGER DEFAULT 0,  -- 被使用次数
    predicted_performance_score DECIMAL(5,4),  -- 模型预测分
    
    -- 来源追踪
    source_type VARCHAR(20) CHECK (source_type IN ('crawled', 'generated', 'manual')),
    source_url TEXT,  -- 若为爬取，记录原视频URL
    crawler_job_id UUID,  -- 关联爬取任务
    
    -- 内容安全
    toxicity_score DECIMAL(5,4),  -- Perspective API
    is_safe BOOLEAN DEFAULT TRUE,
    review_status VARCHAR(20) DEFAULT 'pending' CHECK (review_status IN ('pending', 'approved', 'rejected')),
    
    -- 向量检索
    content_vector_id UUID,  -- Qdrant中的向量ID
    
    -- 审计
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),  -- 创建者：system/user_x
    
    INDEX idx_merchant_platform (merchant_id, platform),
    INDEX idx_performance (predicted_performance_score DESC),
    INDEX idx_hook_type (hook_type, emotion_tag),
    UNIQUE (merchant_id, content_md5)  -- 去重：同一商家相同内容
);

-- 钩子标签关联表（支持多标签）
CREATE TABLE hook_tags (
    tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hook_id UUID REFERENCES hooks(hook_id) ON DELETE CASCADE,
    tag_type VARCHAR(20),  -- type/emotion/length/industry
    tag_value VARCHAR(50),  -- 如"fitness/skincare"
    INDEX idx_hook_id (hook_id),
    INDEX idx_tag_lookup (tag_type, tag_value)
);

-- 钩子效果历史（用于模型训练）
CREATE TABLE hook_performance_history (
    record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hook_id UUID REFERENCES hooks(hook_id),
    video_id UUID,  -- 关联发布的视频
    actual_ctr DECIMAL(5,4),
    actual_completion_rate DECIMAL(5,4),
    actual_share_rate DECIMAL(5,4),
    view_count INTEGER,
    like_count INTEGER,
    comment_count INTEGER,
    collected_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_hook_id (hook_id),
    INDEX idx_collected_at (collected_at)
);

-- 竞品监控源配置
CREATE TABLE competitor_sources (
    source_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    platform VARCHAR(20),
    competitor_handle VARCHAR(100),  -- 竞品账号名
    competitor_user_id VARCHAR(100),  -- 平台用户ID
    is_active BOOLEAN DEFAULT TRUE,
    crawl_frequency INTERVAL DEFAULT '24 hours',
    last_crawl_at TIMESTAMP,
    avg_engagement_rate DECIMAL(5,4),  -- 竞品平均互动率
    INDEX idx_merchant (merchant_id)
);

-- 爬取任务日志
CREATE TABLE crawl_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES competitor_sources(source_id),
    status VARCHAR(20) CHECK (status IN ('running', 'completed', 'failed')),
    crawled_video_count INTEGER,
    new_hooks_found INTEGER,
    error_message TEXT,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    INDEX idx_status (status)
);
```

---

### **1.2 钩子生成引擎**

#### **业务对象**
- **钩子模板(HookTemplate)**: 可参数化的钩子公式
- **生成任务(GenerationTask)**: 单次生成请求
- **生成变体(HookVariant)**: 生成的候选钩子

#### **核心表结构**

```sql
-- 钩子模板库
CREATE TABLE hook_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(100) UNIQUE,  -- 如"Numbered_Reasons"
    template_formula TEXT NOT NULL,  -- "[数字]个原因[动作]"
    platform VARCHAR(20),
    hook_type VARCHAR(20),
    cultural_market VARCHAR(10),  -- US/UK/CA/AU
    usage_count INTEGER DEFAULT 0,
    avg_effectiveness DECIMAL(5,4),  -- 模板历史效果
    is_active BOOLEAN DEFAULT TRUE,
    parameters JSONB  -- 参数定义：[{"name": "数字", "type": "int", "range": "3-10"}]
);

-- 生成任务记录
CREATE TABLE hook_generation_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    product_id UUID,  -- 关联产品
    request_params JSONB,  -- 输入参数：卖点、痛点、风格偏好
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    llm_model_used VARCHAR(50),  -- gpt-4/gemini-pro
    cost USD_DECIMAL,  -- 调用成本
    INDEX idx_merchant_status (merchant_id, status)
);

-- 生成变体表
CREATE TABLE hook_variants (
    variant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES hook_generation_tasks(task_id),
    hook_id UUID REFERENCES hooks(hook_id),  -- 入库后的钩子ID
    rank INTEGER,  -- 生成排序
    content TEXT NOT NULL,
    predicted_ctr DECIMAL(5,4),
    predicted_completion_rate DECIMAL(5,4),
    sentiment_polarity DECIMAL(5,4),  -- -1~1
    brand_safety_score DECIMAL(5,4),  -- 品牌安全分
    is_selected BOOLEAN DEFAULT FALSE,  -- 是否被选中使用
    INDEX idx_task (task_id)
);

-- LLM Prompt版本管理
CREATE TABLE llm_prompt_versions (
    prompt_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module VARCHAR(50),  -- hook_factory/script_generator
    platform VARCHAR(20),
    version VARCHAR(20),  -- v1.2.3
    prompt_template TEXT NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    performance_metrics JSONB  -- {avg_score: 0.85, usage: 120}
);

-- 文化参数配置表
CREATE TABLE cultural_params (
    market_code VARCHAR(10) PRIMARY KEY,  -- US, UK, BR
    emoji_density_range INT[2],  -- [5, 15]
    slang_keywords TEXT[],  -- 俚语列表
    forbidden_symbols TEXT[],  -- 禁忌符号
    tone_preferences JSONB,  -- {formality: "casual", humor_level: 0.7}
    example_hooks TEXT[]
);
```

---

### **1.3 钩子效果预测器**

#### **业务对象**
- **预测模型版本(PredictionModel)**
- **特征工程配置(FeatureConfig)**
- **预测记录(PredictionRecord)**

#### **核心表结构**

```sql
-- 预测模型版本管理
CREATE TABLE prediction_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100),
    version VARCHAR(20),
    algorithm VARCHAR(50),  -- xgboost/lgbm
    training_data_range TSRANGE,  -- 训练数据时间范围
    performance_metrics JSONB,  -- {mse: 0.02, r2: 0.85, auc: 0.92}
    feature_importance JSONB,  -- {hook_length: 0.35, emotion_surprise: 0.28}
    is_deployed BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP,
    retrain_cron VARCHAR(50) DEFAULT '0 2 * * 0'  -- 每周重训
);

-- 特征工程配置表
CREATE TABLE feature_configurations (
    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES prediction_models(model_id),
    feature_name VARCHAR(100),  -- hook_length
    feature_type VARCHAR(20) CHECK (feature_type IN ('numeric', 'categorical', 'embedding')),
    extraction_logic TEXT,  -- SQL或Python代码
    is_active BOOLEAN DEFAULT TRUE,
    weight DECIMAL(5,4)  -- 特征权重
);

-- 预测记录表（用于模型监控）
CREATE TABLE hook_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES prediction_models(model_id),
    hook_id UUID REFERENCES hooks(hook_id),
    predicted_ctr DECIMAL(5,4),
    predicted_completion_rate DECIMAL(5,4),
    confidence_interval JSONB,  -- {lower: 0.05, upper: 0.12}
    actual_ctr DECIMAL(5,4),  -- 回填实际值
    residual DECIMAL(5,4),  -- 预测误差
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_model_hook (model_id, hook_id),
    INDEX idx_residual (ABS(residual) DESC)  -- 监控大误差
);

-- A/B测试回流数据表
CREATE TABLE ab_test_feedback (
    feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID,  -- 关联A/B测试
    hook_variant_id UUID,
    actual_performance JSONB,  -- 实际效果
    feedback_type VARCHAR(20) CHECK (feedback_type IN ('positive', 'negative', 'neutral')),
    model_update_applied BOOLEAN DEFAULT FALSE,  -- 是否已用于模型更新
    INDEX idx_test (test_id)
);
```

---

## **二、脚本生成子模块体系数据库设计**

### **2.1 脚本结构引擎**

#### **业务对象**
- **脚本(Script)**: 完整视频脚本
- **场景(Scene)**: 脚本中的单个镜头
- **叙事模板(NarrativeTemplate)**

#### **核心表结构**

```sql
-- 脚本主表
CREATE TABLE scripts (
    script_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    hook_id UUID REFERENCES hooks(hook_id),  -- 关联钩子
    content_type VARCHAR(30) CHECK (content_type IN ('tutorial', 'story', 'review', 'pain_point', 'interaction')),
    platform VARCHAR(20),
    duration_seconds INTEGER CHECK (duration_seconds <= 90),
    narrative_arc VARCHAR(50),  -- PAS/HeroJourney
    total_scenes INTEGER,
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'review', 'approved', 'rejected')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant (merchant_id),
    INDEX idx_hook (hook_id)
);

-- 场景明细表
CREATE TABLE script_scenes (
    scene_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id UUID REFERENCES scripts(script_id) ON DELETE CASCADE,
    scene_number INTEGER NOT NULL,  -- 场景序号
    duration_seconds INTEGER,
    purpose VARCHAR(50),  -- hook/body/cta/testimonial
    content_type VARCHAR(30),
    visual_description TEXT,  -- AI生成的视觉描述
    copy_text TEXT,  -- 场景文案
    subtitle_timestamp JSONB,  -- {start: 3.2, end: 8.5}
    product_mention JSONB,  -- {product_id: xxx, mention_type: "show"}
    INDEX idx_script (script_id, scene_number)
);

-- 叙事模板库
CREATE TABLE narrative_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(100) UNIQUE,  -- "PAS_v2"
    arc_type VARCHAR(50),  -- PAS/AIDA/4U
    scene_structure JSONB,  -- [{purpose: "hook", duration_ratio: 0.15}, ...]
    platform_constraints JSONB,  -- {instagram_reels_max_duration: 30}
    recommended_use_case TEXT,
    effectiveness_score DECIMAL(5,4),
    is_active BOOLEAN DEFAULT TRUE
);

-- 脚本-产品关联表
CREATE TABLE script_products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id UUID REFERENCES scripts(script_id),
    product_id UUID,
    priority_rank INTEGER,  -- 卖点排序
    mention_scene_numbers INT[],  -- 在哪些场景提及
    key_message TEXT  -- 核心卖点文案
);
```

---

### **2.2 内容填充引擎**

#### **业务对象**
- **用户证言(UserTestimonial)**
- **场景图片推荐(SceneImageRecommendation)**
- **卖点库(ProductSellingPoint)**

#### **核心表结构**

```sql
-- 用户证言池
CREATE TABLE user_testimonials (
    testimonial_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    product_id UUID,
    platform VARCHAR(20),
    original_comment TEXT NOT NULL,
    cleaned_comment TEXT,  -- 清洗后
    user_handle VARCHAR(100),
    engagement_metrics JSONB,  -- {likes: 120, replies: 5}
    sentiment_score DECIMAL(5,4),
    relevance_score DECIMAL(5,4),  -- 与产品相关性（向量检索）
    is_approved BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    vector_id UUID  -- Qdrant向量ID
);

-- 场景图片资源表
CREATE TABLE scene_image_assets (
    asset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID,
    source VARCHAR(50),  -- unsplash/pexels/ugc/upload
    unsplash_id VARCHAR(100),  -- 源平台ID
    image_url TEXT NOT NULL,
    local_uri TEXT,  -- 处理后本地路径
    description_vector_id UUID,  -- Qdrant向量ID
    metadata JSONB,  -- {width: 1080, height: 1920, color_palette: ["#FF6B6B"]}
    license_type VARCHAR(20),  -- free/premium/ugc
    is_processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_vector (description_vector_id)
);

-- 产品卖点库
CREATE TABLE product_selling_points (
    point_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id UUID NOT NULL,
    point_text TEXT NOT NULL,
    priority_score INTEGER,  -- 动态优先级
    target_persona VARCHAR(50),  -- fitness_mom/traveler
    evidence_type VARCHAR(20),  -- data/testimonial/certification
    evidence_data JSONB,  -- {source: "FDA", value: "BPA Free"}
    is_dynamic BOOLEAN DEFAULT FALSE,  -- 是否根据用户画像动态调整
    INDEX idx_product (product_id)
);
```

---

### **2.3 脚本效果分析器**

#### **业务对象**
- **视频播放留存数据(VideoRetention)**
- **场景流失分析(SceneDropoff)**
- **热点标记(HeatPoint)**

#### **核心表结构**

```sql
-- 视频秒级留存数据
CREATE TABLE video_retention_data (
    data_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL,
    merchant_id UUID NOT NULL,
    script_id UUID REFERENCES scripts(script_id),
    second_number INTEGER,  -- 第几秒
    viewer_count INTEGER,  -- 该秒观看人数
    dropoff_count INTEGER,  -- 该秒流失人数
    rewatch_count INTEGER,  -- 重播次数
    collected_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_video_second (video_id, second_number),
    INDEX idx_dropoff (dropoff_count DESC)
);

-- 场景流失分析结果
CREATE TABLE scene_dropoff_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id UUID REFERENCES scripts(script_id),
    scene_id UUID REFERENCES script_scenes(scene_id),
    dropoff_rate DECIMAL(5,4),  -- 该场景流失率
    likely_cause TEXT,  -- AI分析原因："文案过长/画面单调"
    copy_length_penalty DECIMAL(5,4),
    visual_interest_score DECIMAL(5,4),  -- 视觉兴趣度
    improvement_suggestion JSONB,  -- {action: "shorten_copy", target_length: 80}
    created_at TIMESTAMP DEFAULT NOW()
);

-- 热点标记表
CREATE TABLE heat_points (
    point_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID,
    script_id UUID,
    scene_id UUID,
    heat_type VARCHAR(20) CHECK (heat_type IN ('rewatch', 'pause', 'share')),
    heat_score DECIMAL(5,4),
    timestamp_start DECIMAL(6,2),
    timestamp_end DECIMAL(6,2),
    extracted_value TEXT,  -- 提取的高价值文案/画面描述
    INDEX idx_heat_score (heat_score DESC)
);

-- 眼动追踪热图（如有）
CREATE TABLE eye_tracking_heatmaps (
    map_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID,
    scene_id UUID,
    heatmap_image_url TEXT,
    fixation_points JSONB,  -- [{x: 120, y: 340, duration: 2.3}, ...]
    visual_attention_score DECIMAL(5,4),  -- 视觉焦点得分
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## **三、视觉设计子模块体系数据库设计**

### **3.1 视觉资源管理器**

#### **业务对象**
- **视觉资源(VisualAsset)**
- **品牌规范(BrandGuideline)**
- **版权检测记录(CopyrightCheck)**

#### **核心表结构**

```sql
-- 视觉资源主表
CREATE TABLE visual_assets (
    asset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    asset_type VARCHAR(20) CHECK (asset_type IN ('product_image', 'scene_image', 'logo', 'overlay')),
    source_url TEXT,  -- 原始URL
    processed_uri TEXT,  -- 处理后路径
    resolution VARCHAR(20),  -- 1080x1920
    file_format VARCHAR(10),
    file_size_bytes BIGINT,
    
    -- 智能处理字段
    is_background_removed BOOLEAN DEFAULT FALSE,
    enhanced_version_uri TEXT,
    style_transfer_version_uri TEXT,
    color_palette JSONB,  -- 提取的色彩体系
    
    -- 版权信息
    license_type VARCHAR(20),
    copyright_check_status VARCHAR(20) DEFAULT 'pending',
    tineye_match_id VARCHAR(100),
    is_risk_free BOOLEAN DEFAULT FALSE,
    
    metadata JSONB,  -- {product_id: xxx, scene_type: "kitchen"}
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant_type (merchant_id, asset_type),
    INDEX idx_copyright (copyright_check_status)
);

-- 品牌规范库
CREATE TABLE brand_guidelines (
    guideline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID UNIQUE NOT NULL,
    primary_color VARCHAR(7),
    secondary_color VARCHAR(7),
    font_primary VARCHAR(50),
    font_secondary VARCHAR(50),
    logo_asset_id UUID REFERENCES visual_assets(asset_id),
    watermark_position VARCHAR(20),  -- top-left/bottom-right
    watermark_opacity DECIMAL(3,2),
    visual_tone VARCHAR(20),  -- professional/casual/luxury
    do_not_use_elements TEXT[],  -- 禁用元素
    created_at TIMESTAMP DEFAULT NOW()
);

-- 版权检测结果
CREATE TABLE copyright_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID REFERENCES visual_assets(asset_id),
    checker VARCHAR(20),  -- tineye/google
    match_found BOOLEAN,
    match_details JSONB,  -- {similarity: 0.95, source: "shutterstock"}
    risk_level VARCHAR(10) CHECK (risk_level IN ('high', 'medium', 'low')),
    recommended_action VARCHAR(20),  -- replace/approve
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_asset (asset_id)
);
```

---

### **3.2 视觉合成引擎**

#### **业务对象**
- **版式模板(LayoutTemplate)**
- **合成任务(CompositionTask)**
- **合成结果(CompositionResult)**

#### **核心表结构**

```sql
-- 版式模板库（Canva式）
CREATE TABLE layout_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(100) UNIQUE,
    platform VARCHAR(20),
    content_type VARCHAR(20),  -- reels_cover/carousel/feed_post
    aspect_ratio VARCHAR(10),  -- 9:16/1:1/4:5
    preview_image_url TEXT,
    
    -- 模板结构（JSON描述图层）
    layer_structure JSONB,  -- [{type: "text", position: [x,y,w,h], style: {}}]
    dynamic_fields TEXT[],  -- 可替换字段：["headline", "product_image"]
    
    -- 品牌适配规则
    auto_brand_injection JSONB,  -- {logo: {position: "bottom-right"}, color: "primary"}
    font_scaling_rules JSONB,  -- {max_chars: 50, min_font_size: 24}
    
    usage_count INTEGER DEFAULT 0,
    avg_render_time_ms INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(20)  -- system/designer_xxx
);

-- Figma插件上传记录
CREATE TABLE figma_template_uploads (
    upload_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    figma_file_key VARCHAR(100),
    figma_node_id VARCHAR(100),
    template_id UUID REFERENCES layout_templates(template_id),
    designer_id UUID,
    upload_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 合成任务队列
CREATE TABLE composition_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    script_id UUID REFERENCES scripts(script_id),
    template_id UUID REFERENCES layout_templates(template_id),
    
    -- 输入参数
    input_assets JSONB,  -- {product_image: "uuid", background: "uuid"}
    text_content JSONB,  -- {headline: "文案", cta: "Shop Now"}
    canvas_dimensions VARCHAR(20),
    
    -- 任务状态
    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN ('queued', 'rendering', 'completed', 'failed')),
    priority INTEGER DEFAULT 5,  -- 1-10
    
    -- 成本与性能
    render_engine VARCHAR(20),  -- remotion/capcut/aws_batch
    cost_usd DECIMAL(10,4),
    render_time_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    
    -- 输出结果
    output_url TEXT,
    quality_check_score DECIMAL(5,4),
    
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_status_priority (status, priority DESC),
    INDEX idx_merchant (merchant_id)
);

-- 渲染失败日志
CREATE TABLE render_failures (
    failure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES composition_tasks(task_id),
    error_code VARCHAR(50),
    error_message TEXT,
    stack_trace TEXT,
    api_response JSONB,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### **3.3 渲染调度器**

#### **业务对象**
- **渲染成本统计(RenderCost)**
- **渲染引擎配额(RenderQuota)**

#### **核心表结构**

```sql
-- 渲染成本统计（按商家/天）
CREATE TABLE daily_render_costs (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    date DATE NOT NULL,
    render_engine VARCHAR(20),
    total_tasks INTEGER,
    total_cost_usd DECIMAL(10,2),
    avg_cost_per_task DECIMAL(10,4),
    cost_breakdown JSONB,  -- {remotion: 12.5, capcut: 8.3}
    UNIQUE (merchant_id, date, render_engine),
    INDEX idx_date_cost (date, total_cost_usd DESC)
);

-- 渲染引擎配额管理
CREATE TABLE render_engine_quotas (
    engine_name VARCHAR(50) PRIMARY KEY,
    provider VARCHAR(50),  -- aws/remotion
    cost_per_minute DECIMAL(10,4),
    max_concurrent_tasks INTEGER,
    current_month_usage_usd DECIMAL(10,2) DEFAULT 0,
    alert_threshold DECIMAL(5,4) DEFAULT 0.8,  -- 80%用量告警
    is_fallback BOOLEAN DEFAULT FALSE  -- 是否为备用引擎
);
```

---

## **四、字幕配音子模块体系数据库设计**

### **4.1 语音处理核心**

#### **业务对象**
- **语音合成任务(VoiceTask)**
- **音色配置(VoiceProfile)**
- **音频质量检测(AudioQuality)**

#### **核心表结构**

```sql
-- 音色配置表
CREATE TABLE voice_profiles (
    voice_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(20),  -- elevenlabs/polly/playht
    voice_name VARCHAR(100),
    voice_model VARCHAR(50),
    gender VARCHAR(10),
    language VARCHAR(10),
    language_code VARCHAR(10),  -- en-US/en-GB
    age_group VARCHAR(20),  -- young/adult/senior
    supported_emotions TEXT[],  -- [neutral, cheerful, urgent]
    cost_per_1000_chars DECIMAL(10,4),
    quality_score DECIMAL(5,4),  -- 主观评分
    is_default BOOLEAN DEFAULT FALSE,
    sample_audio_url TEXT,
    merchant_id UUID,  -- NULL表示平台公共音色，有值表示商家克隆音色
    INDEX idx_provider (provider, voice_name)
);

-- 语音合成任务
CREATE TABLE voice_synthesis_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id UUID REFERENCES scripts(script_id),
    scene_id UUID REFERENCES script_scenes(scene_id),
    voice_id UUID REFERENCES voice_profiles(voice_id),
    
    input_text TEXT NOT NULL,
    output_audio_url TEXT,
    
    -- 参数配置
    speech_speed_wpm INTEGER,  -- 120/150
    emotion_tag VARCHAR(20),
    stability FLOAT,  -- ElevenLabs参数
    style_exaggeration FLOAT,
    
    -- 状态与成本
    status VARCHAR(20) DEFAULT 'queued',
    cost_usd DECIMAL(10,4),
    duration_ms INTEGER,  -- 音频时长
    
    -- 质量检测
    quality_check JSONB,  -- {snr: 32.5, lufs: -16.2, clarity: "pass"}
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_script (script_id)
);

-- 商家克隆音色样本
CREATE TABLE voice_cloning_samples (
    sample_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    voice_id UUID REFERENCES voice_profiles(voice_id),
    sample_audio_url TEXT NOT NULL,
    sample_text TEXT,  -- 样本文案
    consent_status VARCHAR(20),  -- 语音授权状态
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant (merchant_id)
);
```

---

### **4.2 字幕处理核心**

#### **业务对象**
- **字幕文件(CaptionFile)**
- **字幕样式样式(CaptionStyle)**
- **ASR校对记录(ASRCorrection)**

#### **核心表结构**

```sql
-- 字幕样式模板
CREATE TABLE caption_styles (
    style_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    style_name VARCHAR(100) UNIQUE,
    merchant_id UUID,  -- NULL为平台默认样式
    
    -- 视觉样式
    font_family VARCHAR(50),
    font_size INTEGER,
    font_color VARCHAR(7),
    background_color VARCHAR(7),
    stroke_width INTEGER,
    stroke_color VARCHAR(7),
    
    -- 动态效果
    animation_type VARCHAR(20),  -- pop/fade/slide
    keyword_highlight_color VARCHAR(7),
    highlight_rules JSONB,  -- {keywords: ["free", "new"], color: "#FFD700"}
    
    -- 平台适配
    platform VARCHAR(20),
    constraints JSONB,  -- {max_chars_per_line: 32, max_lines: 2}
    
    is_active BOOLEAN DEFAULT TRUE,
    preview_image_url TEXT
);

-- 字幕文件表
CREATE TABLE caption_files (
    caption_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL,
    task_id UUID REFERENCES voice_synthesis_tasks(task_id),
    
    file_format VARCHAR(10) CHECK (file_format IN ('srt', 'ass', 'vtt')),
    file_url TEXT,
    is_hardcoded BOOLEAN DEFAULT FALSE,  -- 是否硬编码到视频
    
    -- 字幕内容
    language VARCHAR(10),
    subtitle_segments JSONB,  -- [{text: "Hello", start: 0.5, end: 2.1}, ...]
    segment_count INTEGER,
    
    -- ASR校对
    asr_raw_output JSONB,  -- Whisper原始输出
    correction_needed BOOLEAN,
    correction_log JSONB,  -- 修正记录：品牌名、术语
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_video (video_id)
);

-- 术语表（确保一致性）
CREATE TABLE glossary_terms (
    term_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    source_term VARCHAR(100) NOT NULL,
    target_term VARCHAR(100) NOT NULL,
    language VARCHAR(10),
    context VARCHAR(50),  -- product_name/feature
    is_proper_noun BOOLEAN DEFAULT FALSE,
    INDEX idx_merchant_term (merchant_id, source_term)
);
```

---

### **4.3 本地化管理器**

#### **业务对象**
- **本地化包(LocalizationPackage)**
- **文化规则(CulturalRule)**
- **合规要求(ComplianceRequirement)**

#### **核心表结构**

```sql
-- 本地化包主表
CREATE TABLE localization_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    master_content_id UUID,  -- 原始内容ID（脚本/视频）
    content_type VARCHAR(20),  -- script/video/caption
    
    target_market VARCHAR(10),  -- BR/US/DE
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'translating', 'reviewing', 'completed')),
    
    -- 翻译进度
    total_segments INTEGER,
    completed_segments INTEGER,
    translation_cost_usd DECIMAL(10,2),
    
    -- 文化适配
    cultural_adaptations JSONB,  -- {emoji_replaced: 3, slang_changed: 2}
    compliance_violations JSONB,  -- 违规项
    
    -- 最终输出
    localized_content_url TEXT,
    quality_score DECIMAL(5,4),
    
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    INDEX idx_content (master_content_id, target_market)
);

-- 文化禁忌规则库
CREATE TABLE cultural_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_code VARCHAR(10) NOT NULL,
    rule_type VARCHAR(20) CHECK (rule_type IN ('symbol', 'color', 'phrase', 'gesture')),
    forbidden_pattern TEXT NOT NULL,
    severity VARCHAR(10) CHECK (severity IN ('high', 'medium', 'low')),
    suggested_replacement TEXT,
    description TEXT,
    INDEX idx_market (market_code, rule_type)
);

-- 合规要求模板
CREATE TABLE compliance_requirements (
    requirement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_code VARCHAR(10),
    regulation_type VARCHAR(50),  -- GDPR/CCPA/LGPD
    required_disclaimers TEXT[],  -- 必须包含的声明
    privacy_policy_url TEXT,
    data_retention_days INTEGER,
    is_applicable BOOLEAN DEFAULT TRUE
);

-- 价格/单位转换表
CREATE TABLE unit_conversion_rules (
    conversion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_unit VARCHAR(20),
    to_unit VARCHAR(20),
    conversion_factor DECIMAL(20,10),
    market_code VARCHAR(10),
    rounding_rule VARCHAR(20)  -- round/floor/ceil
);
```

---

## **五、A/B测试包子模块体系数据库设计**

### **5.1 测试设计引擎**

#### **业务对象**
- **AB测试(ABTest)**
- **测试变量(TestVariable)**
- **测试分组(TestGroup)**

#### **核心表结构**

```sql
-- A/B测试主表
CREATE TABLE ab_tests (
    test_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    test_name VARCHAR(200),
    content_id UUID,  -- 关联脚本/视频
    
    -- 测试目标
    test_objective VARCHAR(50) CHECK (test_objective IN ('ctr', 'completion', 'conversion', 'engagement')),
    baseline_value DECIMAL(5,4),  -- 基线值
    
    -- 分组策略
    split_method VARCHAR(20) CHECK (split_method IN ('uniform', 'bayesian')),
    group_count INTEGER,
    minimum_sample_size INTEGER,  -- 最小样本量
    
    -- 成功标准
    success_metric VARCHAR(50),
    significance_threshold DECIMAL(3,2) DEFAULT 0.95,
    minimum_detectable_effect DECIMAL(5,4),
    
    -- 状态管理
    status VARCHAR(20) DEFAULT 'design' CHECK (status IN ('design', 'running', 'paused', 'completed', 'terminated')),
    start_at TIMESTAMP,
    end_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant_status (merchant_id, status)
);

-- 测试变量定义
CREATE TABLE test_variables (
    variable_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES ab_tests(test_id) ON DELETE CASCADE,
    
    variable_type VARCHAR(20) CHECK (variable_type IN ('hook', 'visual', 'audio', 'cta', 'caption_style')),
    variable_name VARCHAR(100),  -- "hook_variant_a"
    variable_config JSONB,  -- {hook_id: xxx, template_id: yyy}
    
    -- 智能推荐依据
    predicted_lift DECIMAL(5,4),
    historical_performance JSONB,
    recommendation_reason TEXT,
    INDEX idx_test (test_id)
);

-- 测试分组
CREATE TABLE test_groups (
    group_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES ab_tests(test_id) ON DELETE CASCADE,
    
    group_name VARCHAR(50),  -- A/B/C
    group_type VARCHAR(10) CHECK (group_type IN ('control', 'variant')),
    traffic_allocation_percent DECIMAL(5,2),  -- 40.00
    
    -- 定向规则
    targeting_rules JSONB,  -- {age_range: [18, 24], gender: "female"}
    
    actual_traffic_count INTEGER DEFAULT 0,
    is_winner BOOLEAN,
    INDEX idx_test (test_id)
);

-- 测试冲突检测日志
CREATE TABLE test_conflict_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    test_id UUID,
    conflict_test_id UUID,
    conflict_type VARCHAR(20),  -- overlapping_audience/variable_conflict
    resolution_action VARCHAR(20),  -- reschedule/pause
    detected_at TIMESTAMP DEFAULT NOW()
);
```

---

### **5.2 流量分配器**

#### **业务对象**
- **流量事件(TrafficEvent)**
- **用户分组映射(UserGroupMap)**
- **动态调权日志(DynamicWeightLog)**

#### **核心表结构**

```sql
-- 用户分组映射（确保一致性）
CREATE TABLE user_group_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES ab_tests(test_id),
    user_identifier VARCHAR(100),  -- 设备指纹或用户ID
    
    group_id UUID REFERENCES test_groups(group_id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    
    -- 防污染机制
    assignment_source VARCHAR(20),  -- ip_cookie/device_id
    is_persistent BOOLEAN DEFAULT TRUE,  -- 是否持久化
    
    UNIQUE (test_id, user_identifier),
    INDEX idx_user (user_identifier),
    INDEX idx_test (test_id)
);

-- 实时流量事件（写入Redis，定期归档到PG）
CREATE TABLE traffic_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID,
    group_id UUID,
    user_identifier VARCHAR(100),
    
    event_type VARCHAR(20),  -- impression/play/complete/click
    event_timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB,  -- {device: "mobile", geo: "US"}
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_test_event (test_id, event_type),
    INDEX idx_timestamp (event_timestamp)
);

-- 动态调权日志
CREATE TABLE dynamic_weight_adjustments (
    adjustment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES ab_tests(test_id),
    
    from_allocation JSONB,  -- {A: 0.33, B: 0.33, C: 0.34}
    to_allocation JSONB,
    
    performance_snapshot JSONB,  -- 调权时各组表现
    algorithm_reasoning TEXT,  -- 算法决策原因
    
    adjusted_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_test (test_id)
);
```

---

### **5.3 结果分析器**

#### **业务对象**
- **测试结果(TestResult)**
- **细分分析(SegmentAnalysis)**

#### **核心表结构**

```sql
-- 测试结果表
CREATE TABLE ab_test_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES ab_tests(test_id) UNIQUE,
    
    winning_group_id UUID REFERENCES test_groups(group_id),
    confidence_level DECIMAL(3,2),
    p_value DECIMAL(5,4),
    
    -- 贝叶斯分析
    bayesian_probabilities JSONB,  -- {A: 0.05, B: 0.95}
    
    -- ROI分析
    roi_lift_percent DECIMAL(7,2),
    estimated_annual_impact_usd DECIMAL(12,2),
    
    -- 最终建议
    recommendation_text TEXT,
    recommended_action VARCHAR(20),  -- implement_winner/continue_test/rollback
    
    analysis_completed_at TIMESTAMP DEFAULT NOW()
);

-- 细分分析表
CREATE TABLE segment_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES ab_tests(test_id),
    
    segment_dimension VARCHAR(20),  -- device/geo/time/age
    segment_value VARCHAR(50),  -- mobile/US/night
    
    group_performance JSONB,  -- {A: {ctr: 0.05, conv: 0.02}, B: {...}}
    interaction_effect BOOLEAN,  -- 是否存在交互效应
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_test_dimension (test_id, segment_dimension)
);
```

---

## **六、业务场景服务层数据库设计**

### **场景1: 新账号冷启动服务**

```sql
-- 冷启动任务表
CREATE TABLE cold_start_missions (
    mission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    account_id VARCHAR(100),
    
    -- 任务配置
    start_date DATE,
    duration_days INTEGER DEFAULT 30,
    creativity_level DECIMAL(3,2),
    target_fan_cost DECIMAL(10,2),
    
    -- 执行状态
    current_day INTEGER DEFAULT 1,
    direction_locked BOOLEAN DEFAULT FALSE,
    locked_direction VARCHAR(50),  -- 确定的内容方向
    
    -- 阶段性成果
    daily_follower_growth INTEGER[],
    avg_completion_rate DECIMAL(5,4),
    
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed')),
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant (merchant_id, status)
);
```

---

### **场景2: 矩阵账号规划服务**

```sql
-- 矩阵账号关系图
CREATE TABLE account_matrix_graph (
    relationship_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    main_account_id VARCHAR(100),
    sub_account_id VARCHAR(100),
    
    matrix_type VARCHAR(20),  -- scenario/persona/region
    sub_account_position VARCHAR(50),  -- travel_fitness/office
    
    -- 协同规则
    cross_promotion_rules JSONB,  -- {retweet_delay_hours: 2, story_mention: true}
    
    -- 流量互导数据
    traffic_diversion_rate DECIMAL(5,4),  -- 子→主转化率
    last_synergy_score DECIMAL(5,4),
    
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (main_account_id, sub_account_id)
);
```

---

### **场景3: 账号资料自动生成服务**

```sql
-- 账号资料版本
CREATE TABLE profile_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    account_id VARCHAR(100),
    
    -- 生成的内容
    generated_username VARCHAR(100),
    generated_bio TEXT,
    generated_bio_vector_id UUID,
    
    -- SEO评分
    seo_score DECIMAL(5,4),
    keyword_density JSONB,  -- {blender: 0.15, portable: 0.08}
    
    -- A/B测试
    ab_test_id UUID REFERENCES ab_tests(test_id),
    is_winner BOOLEAN,
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_account (account_id)
);
```

---

### **场景4: 产品相关内容生产服务**

```sql
-- 产品内容生产流水线
CREATE TABLE product_content_assembly_lines (
    line_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    product_id UUID,
    
    trigger_type VARCHAR(20),  -- new_launch/restock/update
    trigger_event_id UUID,  -- 关联事件
    
    -- 内容矩阵
    content_packages JSONB,  -- {announcement: {...}, tutorial: {...}}
    
    -- 调度状态
    schedule_strategy VARCHAR(20),  -- staggered_7_days
    current_stage VARCHAR(50),
    completed_packages TEXT[],
    
    -- 库存联动
    inventory_threshold INTEGER,  -- 低于此库存暂停
    auto_pause_on_low_stock BOOLEAN DEFAULT TRUE,
    
    status VARCHAR(20) DEFAULT 'running',
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### **场景5: 互动内容产生服务**

```sql
-- 互动策略配置
CREATE TABLE engagement_strategies (
    strategy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    account_id VARCHAR(100),
    
    current_strategy VARCHAR(20),  -- question/poll/challenge
    
    -- 触发条件
    trigger_conditions JSONB,  -- {account_age_days: "<30", follower_count: "<10000"}
    
    -- 执行日志
    last_engagement_post_id VARCHAR(100),
    last_engagement_rate DECIMAL(5,4),
    
    -- 热词监控
    hot_keywords_monitored TEXT[],
    auto_generate_on_trend BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant (merchant_id)
);
```

---

### **场景6: 评论回复内容产生服务**

```sql
-- 评论分类结果
CREATE TABLE comment_classifications (
    classification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    platform VARCHAR(20),
    
    comment_id VARCHAR(100) UNIQUE,  -- 平台评论ID
    comment_text TEXT,
    commenter_handle VARCHAR(100),
    
    -- AI分类
    classified_category VARCHAR(20),  -- question/praise/complaint/spam/purchase_intent
    confidence_score DECIMAL(5,4),
    
    -- 意图识别（购买意向）
    purchase_intent JSONB,  -- {intent: "price_inquiry", product: "xxx", urgency: "high"}
    
    -- 处理状态
    reply_generated TEXT,
    reply_sent BOOLEAN DEFAULT FALSE,
    is_hot_lead BOOLEAN DEFAULT FALSE,
    
    -- 人工审核
    review_status VARCHAR(20) DEFAULT 'pending' CHECK (review_status IN ('pending', 'approved', 'rejected', 'escalated')),
    reviewed_by VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT NOW(),
    replied_at TIMESTAMP,
    INDEX idx_merchant (merchant_id, classified_category),
    INDEX idx_hot_lead (is_hot_lead, created_at) WHERE is_hot_lead = TRUE
);

-- 预设回复模板库
CREATE TABLE response_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    
    category VARCHAR(20),  -- question/praise/complaint
    sub_intent VARCHAR(50),  -- price_inquiry/shipping
    
    template_text TEXT NOT NULL,  -- "Price is ${{price}}"
    dynamic_fields TEXT[],  -- ["price", "discount_code"]
    
    usage_count INTEGER DEFAULT 0,
    average_response_time_seconds INTEGER,
    conversion_rate DECIMAL(5,4),
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_category (merchant_id, category, sub_intent)
);

-- CRM集成：高意向用户
CREATE TABLE hot_leads (
    lead_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    comment_classification_id UUID REFERENCES comment_classifications(classification_id),
    
    user_handle VARCHAR(100),
    user_profile_json JSONB,
    
    -- 意向信息
    interested_product_id UUID,
    intent_urgency VARCHAR(20),  -- low/medium/high
    estimated_value_usd DECIMAL(10,2),
    
    -- CRM状态
    crm_deal_id VARCHAR(100),  -- Pipedrive Deal ID
    assigned_sales_rep VARCHAR(100),
    deal_status VARCHAR(20),  -- open/won/lost
    
    first_detected_at TIMESTAMP DEFAULT NOW(),
    last_touch_at TIMESTAMP,
    converted_at TIMESTAMP,
    
    INDEX idx_merchant_status (merchant_id, deal_status),
    INDEX idx_urgency (intent_urgency, created_at)
);
```

---

## **七、系统级数据设计**

### **统一数据总线 (Kafka Topics归档)**

```sql
-- Kafka事件归档表
CREATE TABLE kafka_event_archive (
    archive_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_name VARCHAR(100),
    partition INTEGER,
    offset BIGINT,
    
    event_key VARCHAR(100),
    event_value JSONB,
    event_timestamp TIMESTAMP,
    
    archived_at TIMESTAMP DEFAULT NOW(),
    retention_days INTEGER DEFAULT 90,
    
    INDEX idx_topic_offset (topic_name, offset),
    INDEX idx_key (event_key)
);
```

### **配置中心 (AWS AppConfig快照)**

```sql
-- 配置版本快照
CREATE TABLE config_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_name VARCHAR(100),
    version VARCHAR(20),
    
    config_data JSONB,  -- 完整配置
    applied_to_modules TEXT[],  -- 影响的模块
    
    applied_at TIMESTAMP DEFAULT NOW(),
    applied_by VARCHAR(100),
    
    rollback_snapshot_id UUID REFERENCES config_snapshots(snapshot_id),
    INDEX idx_name_version (config_name, version)
);
```

---

## **八、监控与治理数据库设计**

### **模块健康监控**

```sql
-- 模块健康指标
CREATE TABLE module_health_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_name VARCHAR(50),
    metric_timestamp TIMESTAMP DEFAULT NOW(),
    
    -- 性能指标
    request_count INTEGER,
    success_count INTEGER,
    avg_response_time_ms INTEGER,
    p99_response_time_ms INTEGER,
    
    -- 成本指标
    total_cost_usd DECIMAL(10,2),
    cost_per_request DECIMAL(10,4),
    
    -- 质量指标
    output_quality_score DECIMAL(5,4),
    user_satisfaction_rate DECIMAL(5,4),
    
    INDEX idx_module_time (module_name, metric_timestamp)
);

-- 告警记录
CREATE TABLE health_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    module_name VARCHAR(50),
    alert_type VARCHAR(20),  -- success_rate/cost_spike/quality_drop
    
    actual_value DECIMAL(10,4),
    threshold_value DECIMAL(10,4),
    
    action_taken VARCHAR(20),  -- pause/notify/rollback
    resolved_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_unresolved (resolved_at) WHERE resolved_at IS NULL
);
```

### **内容质量治理**

```sql
-- 人工抽检评分
CREATE TABLE content_quality_reviews (
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    content_id UUID,
    content_type VARCHAR(20),  -- script/visual/video
    
    reviewer_id VARCHAR(100),
    quality_score INTEGER CHECK (quality_score BETWEEN 1 AND 5),
    review_notes TEXT,
    
    -- 质量问题标签
    issue_tags TEXT[],  -- ["brand_logo_misplaced", "grammar_error"]
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_merchant_score (merchant_id, quality_score)
);

-- 自动红线拦截记录
CREATE TABLE red_line_violations (
    violation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    content_id UUID,
    
    violation_type VARCHAR(20),  -- brand_word_error/logo_distortion/banned_term
    violation_details JSONB,
    
    auto_action_taken VARCHAR(20),  -- block/regenerate/escalate
    triggered_module VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_type (violation_type)
);
```

### **成本优化**

```sql
-- 每日成本熔断记录
CREATE TABLE cost_circuit_breakers (
    breaker_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_id UUID NOT NULL,
    trigger_date DATE NOT NULL,
    
    budget_limit_usd DECIMAL(10,2),
    actual_cost_usd DECIMAL(10,2),
    
    downgrade_actions JSONB,  -- {model: "gpt-3.5-turbo", reason: "cost_limit"}
    recovered_at TIMESTAMP,  -- 何时恢复高成本模式
    
    UNIQUE (merchant_id, trigger_date)
);

-- 缓存复用统计
CREATE TABLE cache_reuse_stats (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key VARCHAR(200),
    module_name VARCHAR(50),
    
    hit_count INTEGER DEFAULT 0,
    miss_count INTEGER DEFAULT 0,
    saved_cost_usd DECIMAL(10,2),
    
    last_hit_at TIMESTAMP,
    ttl_seconds INTEGER,
    
    INDEX idx_key (cache_key)
);
```

---

## **九、性能优化与索引策略**

### **分区策略**
```sql
-- 按时间分区：大表按月分区
CREATE TABLE traffic_events PARTITION BY RANGE (event_timestamp);

-- 按商家分区：多租户隔离
CREATE TABLE hooks PARTITION BY HASH (merchant_id);
```

### **物化视图**
```sql
-- 预聚合钩子效果
CREATE MATERIALIZED VIEW hook_performance_summary AS
SELECT 
    hook_id,
    AVG(actual_ctr) as avg_ctr,
    AVG(actual_completion_rate) as avg_completion,
    COUNT(*) as sample_size
FROM hook_performance_history
WHERE collected_at > NOW() - INTERVAL '30 days'
GROUP BY hook_id;

CREATE INDEX idx_hook_perf ON hook_performance_summary (avg_ctr DESC);
```

### **扩展JSONB索引**
```sql
-- GIN索引加速JSONB查询
CREATE INDEX idx_hooks_metadata ON hooks USING GIN (metadata);

-- 特定JSON路径索引
CREATE INDEX idx_content_packages ON product_content_assembly_lines 
USING GIN ((content_packages->'announcement'));
```

---

## **十、数据保留策略**

```sql
-- 设置表级TTL
ALTER TABLE traffic_events SET (ttl = '30 days');
ALTER TABLE video_retention_data SET (ttl = '90 days');

-- 归档脚本
CREATE OR REPLACE FUNCTION archive_old_data()
RETURNS void AS $$
BEGIN
    -- 移动90天前的原始事件到冷存储
    INSERT INTO traffic_events_cold
    SELECT * FROM traffic_events 
    WHERE event_timestamp < NOW() - INTERVAL '90 days';
    
    DELETE FROM traffic_events 
    WHERE event_timestamp < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;
```

---

**总结**：此数据库设计采用**混合范式**，核心实体严格遵循3NF，使用JSONB灵活字段支持AI生成内容，通过**外键约束**保证数据一致性，**分区+物化视图**应对大规模查询，**多租户隔离**确保商家数据安全。每个模块的表结构均支持**水平扩展**和**独立演进**，与业务架构的模块化设计保持一致。