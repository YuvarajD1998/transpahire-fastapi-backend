from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.schemas import ProfileUpdateRequest, ProfileResponse
from app.crud.resume_crud import ProfileCRUD

router = APIRouter(prefix="/profiles", tags=["profiles"])

@router.get("/me", response_model=ProfileResponse)
async def get_me(
    user=Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    profile = await ProfileCRUD.get_by_user_id(db, user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return ProfileResponse(
        id=profile.id,
        user_id=profile.user_id,
        first_name=profile.first_name,
        last_name=profile.last_name,
        phone=profile.phone,
        location=profile.location,
        headline=profile.headline,
        bio=profile.bio,
        linkedin_url=profile.linkedin_url,
        github_url=profile.github_url,
        website_url=profile.website_url,
        profile_completeness=profile.profile_completeness,
        created_at=profile.created_at,
        updated_at=profile.updated_at
    )

@router.put("/me", response_model=ProfileResponse)
async def update_me(
    update: ProfileUpdateRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    profile = await ProfileCRUD.get_by_user_id(db, user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    updated_profile = await ProfileCRUD.update_profile(db, profile.id, update)
    if not updated_profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return ProfileResponse(**updated_profile.__dict__)
