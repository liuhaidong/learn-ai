from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List, Optional
from sqlalchemy import text
from app.core.database import get_session
from app.models.models import Interaction, InstagramAccount
from app.schemas.schemas import (
    InteractionCreate,
    InteractionUpdate,
    Interaction,
    InteractionApproveRequest,
    DiscoveryRequest,
)
from app.services.openai_service import openai_service
from app.services.apify_service import apify_service
from app.models.models import InteractionType, InteractionStatus

router = APIRouter(prefix="/growth", tags=["Growth Engine"])


@router.post("/interactions", response_model=Interaction, status_code=status.HTTP_201_CREATED)
async def create_interaction(
    interaction_data: InteractionCreate,
    session: Session = Depends(get_session)
):
    account = session.get(InstagramAccount, interaction_data.instagram_account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instagram account not found"
        )
    
    new_interaction = Interaction(**interaction_data.model_dump())
    session.add(new_interaction)
    session.commit()
    session.refresh(new_interaction)
    
    return Interaction.model_validate(new_interaction)


@router.get("/interactions/{interaction_id}", response_model=Interaction)
async def get_interaction(
    interaction_id: str,
    session: Session = Depends(get_session)
):
    interaction = session.get(Interaction, interaction_id)
    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found"
        )
    return Interaction.model_validate(interaction)


@router.get("/accounts/{account_id}/interactions", response_model=List[Interaction])
async def get_account_interactions(
    account_id: str,
    status_filter: Optional[str] = None,
    session: Session = Depends(get_session)
):
    query = select(Interaction).where(Interaction.instagram_account_id == account_id)
    
    if status_filter:
        query = query.where(Interaction.status == status_filter)
    
    interactions = session.exec(query.order_by(text("created_at DESC"))).all()
    return [Interaction.model_validate(i) for i in interactions]


@router.post("/interactions/approve")
async def approve_interactions(
    request: InteractionApproveRequest,
    session: Session = Depends(get_session)
):
    approved_count = 0
    
    for interaction_id in request.interaction_ids:
        interaction = session.get(Interaction, interaction_id)
        if interaction:
            interaction.status = InteractionStatus.EXECUTED
            session.add(interaction)
            approved_count += 1
    
    session.commit()
    
    return {"approved_count": approved_count}


@router.post("/discover/users")
async def discover_users(
    request: DiscoveryRequest,
    session: Session = Depends(get_session)
):
    account = session.get(InstagramAccount, request.instagram_account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instagram account not found"
        )
    
    try:
        followers = await apify_service.get_followers(
            username=account.username,
            limit=request.limit,
        )
        
        users_to_interact = []
        
        for follower in followers:
            is_private = follower.get("is_private", False)
            if is_private:
                continue
            
            username = follower.get("username", "")
            if not username:
                continue
            
            interaction = Interaction(
                instagram_account_id=request.instagram_account_id,
                target_username=username,
                type=InteractionType.FOLLOW,
                status=InteractionStatus.PENDING_APPROVAL,
            )
            session.add(interaction)
            users_to_interact.append(interaction)
        
        session.commit()
        
        return {
            "discovered_count": len(users_to_interact),
            "users": followers,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to discover users: {str(e)}"
        )


@router.post("/generate/comment")
async def generate_comment(
    post_caption: str,
    context: Optional[str] = None,
    post_image_url: str = ""
):
    try:
        comment = await openai_service.generate_comment(
            post_image_url=post_image_url,
            post_caption=post_caption,
            context=context,
        )
        
        return {"comment": comment}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate comment: {str(e)}"
        )


@router.delete("/interactions/{interaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_interaction(
    interaction_id: str,
    session: Session = Depends(get_session)
):
    interaction = session.get(Interaction, interaction_id)
    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found"
        )
    
    session.delete(interaction)
    session.commit()
    
    return None
