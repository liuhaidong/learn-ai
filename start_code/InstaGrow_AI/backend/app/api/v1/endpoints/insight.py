from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List
from app.core.database import get_session
from app.models.models import Competitor, InstagramAccount
from app.schemas.schemas import CompetitorCreate, Competitor, CompetitorAnalysisRequest
from app.services.apify_service import apify_service
from app.services.openai_service import openai_service

router = APIRouter(prefix="/insight", tags=["Insight Engine"])


@router.post("/competitors", response_model=Competitor, status_code=status.HTTP_201_CREATED)
async def add_competitor(
    competitor_data: CompetitorCreate,
    session: Session = Depends(get_session)
):
    existing_competitor = session.exec(
        select(Competitor).where(
            (Competitor.workspace_id == competitor_data.workspace_id) &
            (Competitor.username == competitor_data.username)
        )
    ).first()
    
    if existing_competitor:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Competitor already exists in this workspace"
        )
    
    new_competitor = Competitor(**competitor_data.model_dump())
    session.add(new_competitor)
    session.commit()
    session.refresh(new_competitor)
    
    return Competitor.model_validate(new_competitor)


@router.get("/competitors/{competitor_id}", response_model=Competitor)
async def get_competitor(
    competitor_id: str,
    session: Session = Depends(get_session)
):
    competitor = session.get(Competitor, competitor_id)
    if not competitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Competitor not found"
        )
    return Competitor.model_validate(competitor)


@router.get("/workspaces/{workspace_id}/competitors", response_model=List[Competitor])
async def get_workspace_competitors(
    workspace_id: str,
    session: Session = Depends(get_session)
):
    competitors = session.exec(
        select(Competitor).where(Competitor.workspace_id == workspace_id)
    ).all()
    return [Competitor.model_validate(c) for c in competitors]


@router.post("/competitors/analyze")
async def analyze_competitors(request: CompetitorAnalysisRequest):
    results = []
    
    for username in request.usernames:
        try:
            profile_data = await apify_service.get_instagram_profile(username)
            
            if profile_data:
                analysis = await openai_service.generate_competitor_analysis(profile_data)
                
                results.append({
                    "username": username,
                    "profile": profile_data,
                    "analysis": analysis,
                })
            else:
                results.append({
                    "username": username,
                    "error": "Profile not found or private",
                })
        except Exception as e:
            results.append({
                "username": username,
                "error": str(e),
            })
    
    return {"results": results}


@router.post("/competitors/discover")
async def discover_competitors(
    keywords: List[str],
    hashtags: List[str] = [],
    limit: int = 10
):
    try:
        competitors = await apify_service.search_competitors(
            keywords=keywords,
            hashtags=hashtags,
            limit=limit
        )
        return {"competitors": competitors}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to discover competitors: {str(e)}"
        )


@router.delete("/competitors/{competitor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_competitor(
    competitor_id: str,
    session: Session = Depends(get_session)
):
    competitor = session.get(Competitor, competitor_id)
    if not competitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Competitor not found"
        )
    
    session.delete(competitor)
    session.commit()
    
    return None
