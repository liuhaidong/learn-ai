"""Instagram Automation - Complete Implementation Guide"""

# Instagram Automation System for E-commerce Cross-Border Sales
# 2 Test Accounts | 6 Product Categories | $100/month Budget
# English Content | OpenAI + Midjourney API Integration

## 📁 Project Structure

```
instagram_automation/
├── .env.example              # Configuration template
├── .env.test                # Test environment config
├── requirements.txt           # Python dependencies
├── docker-compose.yml        # Database and Redis setup
├── main.py                  # Main application entry point
├── config/
│   ├── settings.py          # Environment configuration
│   └── constants.py         # Product categories, hashtags, prompts
├── database/
│   ├── connection.py        # PostgreSQL connection pool
│   ├── models.py            # SQLAlchemy ORM models
│   └── repositories.py       # Data access layer
├── instagram/
│   ├── client.py            # Instagram API wrapper (instagrapi)
│   ├── account_manager.py    # 2-account management
│   └── __init__.py
├── services/
│   ├── content_generator.py  # E-commerce content generation
│   ├── interaction.py        # Daily interaction automation
│   ├── rate_limiter.py       # Rate limiting for 2 accounts
│   └── cost_monitor.py      # Budget tracking ($100/month)
├── ai/
│   ├── openai_client.py      # OpenAI integration (captions, comments)
│   └── midjourney_client.py # Midjourney integration (images)
├── utils/
│   └── logger.py            # Logging system
└── tests/
    ├── unit/                # Unit tests
    └── integration/         # Integration tests
```

## 🎯 6 Product Categories

1. **Charging Cable** (#datacable, #usbcable)
   - Price: $5-30
   - Features: Fast charging, durable, tangle-free

2. **Charger** (#charger, #fastcharging)
   - Price: $10-50
   - Features: Quick charge, compact, safety protection

3. **Earbuds** (#earbuds, #wirelessearbuds)
   - Price: $20-100
   - Features: Noise reduction, long battery, comfortable fit

4. **Screen Protector** (#screenprotector, #temperedglass)
   - Price: $5-20
   - Features: Scratch resistant, easy install, high clarity

5. **Phone Case** (#phonecase, #iphonecase)
   - Price: $10-40
   - Features: Shock absorption, slim design, premium material

6. **Noise-Cancelling Headphone** (#noisecancelling, #headphones)
   - Price: $50-300
   - Features: Active NC, high-resolution audio, long battery

## 💰 Budget Breakdown ($100/month)

```
Total Budget: $100.00/month
Daily Budget: $3.33/day

Allocation:
├── OpenAI API (Text Generation):        $50.00 (50%)
│   ├── Captions: ~$30 (6-7/day @ $0.15/1K tokens)
│   └── Comments: ~$20 (30/day @ $0.15/1K tokens)
│
├── Midjourney API (Image Generation):   $40.00 (40%)
│   └── Images: ~1-2/day @ $0.05/image
│
├── Cloud/Database:                       $8.00  (8%)
│   ├── PostgreSQL: $5.00
│   └── Redis: $3.00
│
└── Proxy (Optional):                       $2.00  (2%)
```

## 🔄 2-Account Configuration

**Account 1 (Phone Accessories)**
- Username: test_phone_case_1 (configure in .env)
- Primary Categories: phone_case, phone_film
- Daily Interactions: 20-25
- Content: 1-2 posts/day

**Account 2 (Audio Accessories)**
- Username: test_earbuds_2 (configure in .env)
- Primary Categories: earbuds, noise_cancelling_headphone
- Daily Interactions: 20-25
- Content: 1-2 posts/day

**Shared Categories** (Rotate between accounts)
- charging_cable
- charger

## 📊 Daily Interaction Distribution

```
Morning (9-11 AM):    30% (15 interactions)
Afternoon (2-4 PM):   30% (15 interactions)
Evening (8-10 PM):     40% (20 interactions)

Total: 50 interactions/day across both accounts
```

**Interaction Types:**
- Likes: 60% (30 likes)
- Comments: 30% (15 comments)
- Follows: 10% (5 follows)

## 📝 Content Generation Strategy

**Content Types:**
- Product Review: 35%
- Value Deal: 25%
- Feature Highlight: 20%
- Comparison: 10%
- Testimonial: 5%
- Installation Guide: 5%

**E-commerce Optimizations:**
- Strong CTA: "Link in bio", "Shop now", "Limited stock"
- Urgency: Scarcity, time-limited offers
- Social proof: "Bestseller", "Customer favorite"
- Shipping: "Free worldwide shipping", "24h delivery"
- Discount: "10-20% OFF" promotions

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create .env file from template
cp .env.example .env

# Edit .env with your credentials
nano .env

# Required:
# - OpenAI API Key
# - Midjourney API Key
# - Instagram test account credentials (2 accounts)
```

### 2. Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python3 -m instagram_automation.database.init_db
```

### 3. Run Application

```bash
# Start the application
python main.py

# Menu Options:
# 1. Generate Content - Create posts for any category
# 2. Run Interactions - Execute daily engagement
# 3. Check Costs - Monitor API usage
# 4. Account Status - View account details
# 5. Exit - Safe logout
```

## 📈 Key Features Implemented

✅ **Multi-Account Management**
   - 2 test accounts with different product focuses
   - Automatic rotation for shared categories
   - Usage tracking and balancing

✅ **6-Category Content Generation**
   - E-commerce focused prompts
   - English captions with sales CTAs
   - Professional product photography via Midjourney
   - Hashtag generation (15-25 tags)

✅ **Daily Interaction Automation**
   - 30-50 interactions/day total
   - Smart rate limiting (account and global)
   - Natural delays (5-30s between actions)
   - AI-generated authentic comments

✅ **Cost Monitoring**
   - Real-time cost tracking
   - Budget alerts (80%, 100% thresholds)
   - Daily reports
   - Monthly summaries

✅ **Safety Features**
   - Rate limiting to avoid bans
   - Human-like interaction patterns
   - Account rotation
   - Session management

## 🔧 Configuration Files

### .env.example
```env
# Instagram Test Accounts
INSTAGRAM_TEST_ACCOUNT_1=test_phone_case_1
INSTAGRAM_TEST_PASSWORD_1=your_password_1
INSTAGRAM_TEST_ACCOUNT_2=test_earbuds_2
INSTAGRAM_TEST_PASSWORD_2=your_password_2

# OpenAI API
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini

# Midjourney API (ImaginePro)
MIDJOURNEY_API_KEY=your-midjourney-key-here

# Database
DATABASE_URL=postgresql://dbuser:password@localhost:5432/instagram_automation

# Budget
MONTHLY_BUDGET=100

# Rate Limits
MAX_LIKES_PER_HOUR=20
MAX_FOLLOWS_PER_HOUR=10
MAX_COMMENTS_PER_HOUR=5
MAX_DAILY_INTERACTIONS=25
```

## 📊 Database Schema

**Core Tables:**
- accounts: Instagram accounts with session data
- product_categories: 6 category configurations
- content_tasks: Generated content tracking
- interactions: Engagement logs
- api_usage_logs: Cost tracking
- system_logs: Debug and monitoring

## ⚠️ Important Notes

1. **Create Test Accounts First**
   - Instagram test accounts should have "test_" prefix
   - Use temporary email services
   - Complete profile setup before automation

2. **API Keys Required**
   - OpenAI: https://platform.openai.com/api-keys
   - Midjourney: https://imaginepro.ai/dashboard
   - Budget: $100/month total

3. **Rate Limits Are Critical**
   - Default: 20 likes/hour, 10 follows/hour
   - 5 comments/hour
   - 25 daily interactions per account
   - 50 total across both accounts

4. **Content Language**
   - All generated content in English
   - Optimized for international audiences
   - E-commerce focused (sales, deals, CTAs)

5. **Budget Control**
   - Monitor daily costs closely
   - Image caching enabled by default
   - Stop if budget exceeded

## 🐛 Troubleshooting

**Issue: "Import error: asyncpg"**
```bash
pip install asyncpg
```

**Issue: "Import error: instagrapi"**
```bash
pip install instagrapi
```

**Issue: Database connection failed**
```bash
# Check PostgreSQL is running
docker-compose up -d

# Verify DATABASE_URL in .env
```

**Issue: API errors**
```bash
# Verify API keys in .env
# Check API status pages
# Review account credits
```

## 📚 Further Development

**Phase 1: Testing (Week 9)**
- [ ] Set up 2 test Instagram accounts
- [ ] Get API keys (OpenAI, Midjourney)
- [ ] Test content generation (all 6 categories)
- [ ] Test interaction automation
- [ ] Verify cost tracking

**Phase 2: Production (Week 10+)**
- [ ] Add real accounts
- [ ] Increase budget if needed
- [ ] Deploy to cloud server
- [ ] Add monitoring dashboard
- [ ] Implement proxy rotation

## 📞 Support & Documentation

- OpenAI Docs: https://platform.openai.com/docs
- Midjourney/ImaginePro: https://docs.imaginepro.ai
- Instagram API: https://github.com/subzeroid/instagrapi
- Project Repository: [Your GitHub URL]

---

**Next Steps:**
1. Copy .env.example to .env
2. Add your API keys and test account credentials
3. Run: `python main.py`
4. Select option 1 to generate test content
5. Monitor costs and adjust as needed

**Cost-Saving Tips:**
- Enable image caching (default: true)
- Batch generate content
- Use gpt-4o-mini (most cost-effective)
- Reuse captions with minor edits
- Monitor daily usage closely

---

**Happy Automating! 🚀**
