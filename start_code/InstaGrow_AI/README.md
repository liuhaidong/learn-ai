# InstaGrow AI

An intelligent AI Agent for automating and optimizing Instagram marketing campaigns for cross-border e-commerce.

## Core Modules

### 1. Insight Engine (市场与竞品洞察模块)
- Competitor discovery and analysis
- Profile strategy analysis
- Content strategy analysis
- Hashtag effectiveness analysis
- Audience profiling

### 2. Content Factory (内容创作与规划模块)
- Content calendar planning
- AI caption generation (multi-style)
- AI visuals generation (DALL-E 3)
- Smart hashtag recommendations

### 3. Growth Engine (用户增长与互动模块)
- Target user discovery
- Intelligent human-like interactions
- AI-powered commenting
- Growth tracking and analytics

## Tech Stack

### Frontend
- Framework: Next.js (React)
- Styling: Tailwind CSS + Shadcn/ui
- State Management: Zustand
- Data Fetching: React Query (TanStack Query)
- Deployment: Vercel

### Backend
- Language: Python 3.11+
- Framework: FastAPI
- ORM: SQLModel
- Deployment: AWS Fargate

### AI Core
- LLM: OpenAI GPT-4o
- Image Generation: DALL-E 3
- Vision: GPT-4o Vision

### Data & External Services
- Database: PostgreSQL (AWS RDS)
- Instagram Data: Apify API
- Container: Docker
- CI/CD: GitHub Actions

## Architecture

```
Frontend (Next.js on Vercel)
    ↓
API Gateway (AWS)
    ↓
FastAPI Backend (AWS Fargate)
    ├── OpenAI API (AI Core)
    ├── Apify API (Instagram Data)
    └── PostgreSQL (AWS RDS)
```

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.11+
- PostgreSQL 15+
- Docker

### Installation

1. Clone the repository
2. Set up the backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Set up the frontend:
   ```bash
   cd frontend
   npm install
   ```
4. Configure environment variables
5. Run the services

## Project Structure

```
InstaGrow_AI/
├── frontend/          # Next.js frontend application
├── backend/           # FastAPI backend application
├── .github/           # GitHub Actions workflows
├── docker/            # Docker configurations
└── README.md
```

## Development

See individual README files in `frontend/` and `backend/` for detailed setup instructions.
