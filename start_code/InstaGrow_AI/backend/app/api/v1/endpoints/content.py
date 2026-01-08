from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List, Optional
from sqlalchemy import text
from app.core.database import get_session
from app.models.models import ContentPiece, InstagramAccount
from app.schemas.schemas import (
    ContentPieceCreate,
    ContentPieceUpdate,
    ContentPiece,
    ContentPieceGenerateRequest,
)
from app.services.openai_service import openai_service

router = APIRouter(prefix="/content", tags=["Content Factory"])


@router.post("/pieces", response_model=ContentPiece, status_code=status.HTTP_201_CREATED)
async def create_content_piece(
    content_data: ContentPieceCreate,
    session: Session = Depends(get_session)
):
    account = session.get(InstagramAccount, content_data.instagram_account_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instagram account not found"
        )
    
    new_content = ContentPiece(**content_data.model_dump())
    session.add(new_content)
    session.commit()
    session.refresh(new_content)
    
    return ContentPiece.model_validate(new_content)


@router.get("/pieces/{content_id}", response_model=ContentPiece)
async def get_content_piece(
    content_id: str,
    session: Session = Depends(get_session)
):
    content = session.get(ContentPiece, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content piece not found"
        )
    return ContentPiece.model_validate(content)


@router.get("/accounts/{account_id}/pieces", response_model=List[ContentPiece])
async def get_account_content(
    account_id: str,
    status_filter: Optional[str] = None,
    session: Session = Depends(get_session)
):
    query = select(ContentPiece).where(ContentPiece.instagram_account_id == account_id)
    
    if status_filter:
        query = query.where(ContentPiece.status == status_filter)
    
    content_pieces = session.exec(query.order_by(text("created_at DESC"))).all()
    return [ContentPiece.model_validate(c) for c in content_pieces]


@router.put("/pieces/{content_id}", response_model=ContentPiece)
async def update_content_piece(
    content_id: str,
    update_data: ContentPieceUpdate,
    session: Session = Depends(get_session)
):
    content = session.get(ContentPiece, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content piece not found"
        )
    
    update_dict = update_data.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        setattr(content, key, value)
    
    session.add(content)
    session.commit()
    session.refresh(content)
    
    return ContentPiece.model_validate(content)


@router.delete("/pieces/{content_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_content_piece(
    content_id: str,
    session: Session = Depends(get_session)
):
    content = session.get(ContentPiece, content_id)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content piece not found"
        )
    
    session.delete(content)
    session.commit()
    
    return None


@router.post("/generate/caption")
async def generate_caption(request: ContentPieceGenerateRequest):
    try:
        account = {
            "product_category": "general",
            "brand_persona": request.brand_tone or "professional",
        }
        
        result = await openai_service.generate_caption(
            product_description=request.product_description,
            brand_tone=request.brand_tone or "professional",
            target_audience=request.target_audience,
            hashtags=True,
        )
        
        return {
            "caption": result.get("caption", ""),
            "hashtags": result.get("hashtags", []),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate caption: {str(e)}"
        )


@router.post("/generate/image")
async def generate_image(
    prompt: str,
    style: str = "professional",
    aspect_ratio: str = "1:1"
):
    try:
        image_url = await openai_service.generate_image(
            prompt=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
        )
        
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate image: {str(e)}"
        )


@router.post("/generate/calendar")
async def generate_calendar(
    product_category: str,
    brand_tone: str = "professional",
    days: int = 7
):
    try:
        calendar = await openai_service.generate_content_calendar(
            product_category=product_category,
            brand_tone=brand_tone,
            days=days,
        )
        
        return {"calendar": calendar}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate content calendar: {str(e)}"
        )
