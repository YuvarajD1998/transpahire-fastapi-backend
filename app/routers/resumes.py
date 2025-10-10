# resumes.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from app.database import get_db
from app.dependencies import get_current_user, require_subscription
from app.models.enums import ParseStatus, SubscriptionTier,SkillSource
from app.models.schemas import CritiqueSection, CritiqueSections, ResumeResponse, ResumeCritiqueResponse, ParsedResumeData, CritiqueData, ResumeCritiqueResponse,SetPrimaryResumeRequest,SetPrimaryResumeResponse
from app.services.file_service import FileService, ResumeParserService
from app.services.critique_service import CritiqueService
from app.utils.file_utils import validate_upload
from app.crud.resume_crud import ProfileCRUD, ResumeCRUD, ResumeCritiqueCRUD
from app.database import db_manager
from app.services.resume_parsing_service import ResumeDataService
from app.services.critique_service import CritiqueService


router = APIRouter(prefix="/resumes", tags=["resumes"])


async def parse_and_process_resume_background(
    resume_id: int, 
    profile_id: int, 
    file_content: bytes, 
    filename: str, 
    enhance_images: bool,
    is_primary: bool  # Add this parameter
):
    """Enhanced background task with conditional embedding generation."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        async for db_session in db_manager.get_session():
            try:
                # Set status to processing
                await ResumeCRUD.update_resume(
                    db_session, 
                    resume_id, 
                    {"parse_status": ParseStatus.PROCESSING}
                )
                await db_session.commit()
                
                # Parse the resume
                parser = ResumeParserService()
                parsed_data = await parser.parse(file_content, filename, enhance_images=enhance_images)
                
                # Convert to dict
                parsed_data_dict = (
                    parsed_data.model_dump() 
                    if hasattr(parsed_data, "model_dump") 
                    else dict(parsed_data)
                )
                
                # Update with parsed data
                await ResumeCRUD.update_resume(
                    db_session, 
                    resume_id, 
                    {
                        "parse_status": ParseStatus.COMPLETED,
                        "parsed_data": parsed_data_dict,
                        "confidence_score": parsed_data_dict.get("confidence_score", 0.85)
                    }
                )
                await db_session.commit()
                
                # CONDITIONAL PROCESSING based on is_primary flag
                if is_primary:
                    # PRIMARY RESUME: Full processing with embeddings
                    print(f"Processing PRIMARY resume {resume_id} with embeddings")
                    resume_service = ResumeDataService()
                    await resume_service.process_parsed_resume_data_with_embeddings(
                        db_session, profile_id, resume_id, parsed_data_dict
                    )
                else:
                    # NON-PRIMARY RESUME: Store only raw+structured JSON, no embeddings, no DB insertion
                    print(f"Storing NON-PRIMARY resume {resume_id} - raw JSON only, no embeddings")
                    # The parsed_data is already stored in the resume record above
                    # No need to insert skills, experience, education into separate tables
                    # No embedding generation
                
                print(f"Successfully processed resume {resume_id} (is_primary={is_primary}) on attempt {attempt + 1}")
                return
                
            except Exception as e:
                await db_session.rollback()
                if attempt == max_retries - 1:
                    await ResumeCRUD.update_resume(
                        db_session, 
                        resume_id, 
                        {"parse_status": ParseStatus.FAILED}
                    )
                    await db_session.commit()
                    print(f"Background parsing failed for resume {resume_id} after {max_retries} attempts: {e}")
                else:
                    print(f"Attempt {attempt + 1} failed for resume {resume_id}: {e}")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                break


@router.post("/upload", response_model=ResumeResponse)
async def upload_resume(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    is_primary: bool = Form(False),  # Changed from query param to form field
    enhance_images: bool = Form(True),
    profile_id: int = Form(...),  # Add these for direct access
    resume_id: Optional[int] = Form(None),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload resume with conditional processing based on is_primary flag."""
    
    from app.utils.file_utils import validate_upload
    validate_upload(file)
    
    # Handle user as dict or object
    user_id = user.get("id") if isinstance(user, dict) else user.id
    
    # Use provided profile_id or fetch by user_id
    if profile_id:
        profile = await ProfileCRUD.get_by_id(db, profile_id)
    else:
        profile = await ProfileCRUD.get_by_user_id(db, user_id)
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    content = await file.read()
    
    fs = FileService()
    file_path = await fs.save_file(content, file.filename, file.content_type)
    
    # Create resume record
    if resume_id:
        # Update existing resume
        resume = await ResumeCRUD.get_resume_by_id(db, resume_id, profile.id)
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        resume = await ResumeCRUD.update_resume(
            db, 
            resume_id, 
            {
                "filename": file.filename.split("/")[-1],
                "original_name": file.filename,
                "file_path": file_path,
                "file_size": len(content),
                "mimetype": file.content_type,
                "is_primary": is_primary,
                "parse_status": ParseStatus.PENDING
            }
        )
    else:
        # Create new resume
        resume = await ResumeCRUD.create_resume(
            db=db,
            profile_id=profile.id,
            filename=file.filename.split("/")[-1],
            original_name=file.filename,
            file_path=file_path,
            file_size=len(content),
            mimetype=file.content_type,
            is_primary=is_primary
        )
    
    # Set as primary if requested
    if is_primary:
        await ResumeCRUD.set_primary_resume(db, profile.id, resume.id)
    
    # Add background task with is_primary flag
    background.add_task(
        parse_and_process_resume_background, 
        resume.id, 
        profile.id,
        content, 
        file.filename, 
        enhance_images,
        is_primary  # Pass the flag
    )
    
    return ResumeResponse(
        id=resume.id,
        filename=resume.filename,
        original_name=resume.original_name,
        file_path=resume.file_path,
        file_size=resume.file_size,
        mimetype=resume.mimetype,
        parse_status=resume.parse_status,
        confidence_score=resume.confidence_score,
        is_primary=resume.is_primary,
        created_at=resume.created_at,
        parsed_data=None
    )


@router.post("/{resume_id}/set-primary", response_model=SetPrimaryResumeResponse)
async def set_resume_as_primary(
    resume_id: int,
    request: SetPrimaryResumeRequest,
    background: BackgroundTasks,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Set a resume as primary and reprocess profile data with new embeddings.
    This will:
    1. Set the resume as primary
    2. Delete OLD AI_EXTRACTED data from profile
    3. Overwrite profile fields with new primary resume's parsed data
    4. Generate fresh embeddings
    5. Insert new structured data into DB
    6. Preserve MANUAL data
    """
    
    # Validate resume exists and belongs to the profile
    resume = await ResumeCRUD.get_resume_by_id(db, resume_id, request.profile_id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    if not resume.parsed_data:
        raise HTTPException(
            status_code=400, 
            detail="Resume has not been parsed yet. Please wait for parsing to complete."
        )
    
    if resume.parse_status != ParseStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Resume parsing is not complete. Current status: {resume.parse_status}"
        )
    
    # Set as primary
    await ResumeCRUD.set_primary_resume(db, request.profile_id, resume_id)
    
    # Process in background to avoid timeout
    background.add_task(
        process_primary_resume_change,
        request.profile_id,
        resume_id,
        resume.parsed_data
    )
    
    return SetPrimaryResumeResponse(
        message="Primary resume is being processed",
        profile_id=request.profile_id,
        resume_id=resume_id,
        embeddings_regenerated=True,
        data_updated=True
    )


async def process_primary_resume_change(
    profile_id: int,
    resume_id: int,
    parsed_data: dict
):
    """
    Background task to process primary resume change.
    - Deletes old AI_EXTRACTED data (preserves MANUAL data)
    - Overwrites profile fields with new primary resume data
    - Generates fresh embeddings
    - Inserts new structured data
    """
    async for db_session in db_manager.get_session():
        try:
            print(f"Processing primary resume change for profile {profile_id}, resume {resume_id}")
            
            # 1. Delete OLD AI_EXTRACTED data (preserve MANUAL data)
            await delete_ai_extracted_data(db_session, profile_id)
            
            # 2. Overwrite profile fields with new parsed data
            await ResumeDataService.update_profile_info(
                db_session, 
                profile_id, 
                parsed_data.get("personal_info", {}),
                parsed_data.get("summary")
            )
            
            # 3. Process and insert new structured data WITH embeddings
            resume_service = ResumeDataService()
            await resume_service.process_parsed_resume_data_with_embeddings(
                db_session, 
                profile_id, 
                resume_id, 
                parsed_data
            )
            
            print(f"Successfully processed primary resume change for profile {profile_id}")
            
        except Exception as e:
            await db_session.rollback()
            print(f"Error processing primary resume change: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


async def delete_ai_extracted_data(db: AsyncSession, profile_id: int):
    """Delete only AI_EXTRACTED data, preserve MANUAL data."""
    from sqlalchemy import delete, and_
    from app.models.database_models import ProfileSkill, WorkExperience, Education
    
    # Delete AI_EXTRACTED skills
    await db.execute(
        delete(ProfileSkill).where(
            and_(
                ProfileSkill.profile_id == profile_id,
                ProfileSkill.source == SkillSource.AI_EXTRACTED
            )
        )
    )
    
    # Delete AI_EXTRACTED work experiences
    await db.execute(
        delete(WorkExperience).where(
            and_(
                WorkExperience.profile_id == profile_id,
                WorkExperience.source == 'AI_EXTRACTED'
            )
        )
    )
    
    # Delete AI_EXTRACTED education
    await db.execute(
        delete(Education).where(
            and_(
                Education.profile_id == profile_id,
                Education.source == 'AI_EXTRACTED'
            )
        )
    )
    
    # Delete old embedding
    from app.models.database_models import Embedding
    await db.execute(
        delete(Embedding).where(
            and_(
                Embedding.entity_type == 'profile',
                Embedding.entity_id == profile_id
            )
        )
    )
    
    await db.commit()
    print(f"Deleted AI_EXTRACTED data for profile {profile_id}")


@router.post("/{resume_id}/critique", response_model=ResumeCritiqueResponse)
async def critique_resume(
    resume_id: int,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_subscription(SubscriptionTier.PREMIUM))
):
    """Generate AI-powered critique for a resume."""
    try:
        profile = await ProfileCRUD.get_by_user_id(db, user["id"])
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        resume = await ResumeCRUD.get_resume_by_id(db, resume_id, profile.id)
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if resume.parse_status != ParseStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Resume not parsed yet or parsing failed")
        
        if not resume.parsed_data:
            raise HTTPException(status_code=400, detail="No parsed data available")
        
        # Generate critique using the service
        critique_service = CritiqueService()
        critique_data_dict = await critique_service.generate_critique(
            ParsedResumeData(**resume.parsed_data)
        )
        
        # Create critique record in database
        critique = await ResumeCritiqueCRUD.create_critique(
            db=db,
            resume_id=resume.id,
            critique_data=critique_data_dict
        )
        
        # Convert dict to Pydantic model for response validation
        critique_data_model = CritiqueData(**critique_data_dict)
        
        return ResumeCritiqueResponse(
            id=critique.id,
            resume_id=critique.resume_id,
            critique_data=critique_data_model,
            created_at=critique.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating critique for resume {resume_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to generate resume critique. Please try again later."
        )



@router.get("/{resume_id}/critiques", response_model=List[ResumeCritiqueResponse])
async def get_resume_critiques(
    resume_id: int,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    profile = await ProfileCRUD.get_by_user_id(db, user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    resume = await ResumeCRUD.get_resume_by_id(db, resume_id, profile.id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    critiques = await ResumeCritiqueCRUD.get_critiques_by_resume(db, resume_id)
    
    return [
        ResumeCritiqueResponse(
            id=critique.id,
            resume_id=critique.resume_id,
            critique_data=CritiqueData(
                overall_score=critique.overall_score,
                sections=CritiqueSections(**critique.sections) if critique.sections else CritiqueSections(
                    experience=CritiqueSection(score=0, feedback="No feedback"),
                    skills=CritiqueSection(score=0, feedback="No feedback"),
                    education=CritiqueSection(score=0, feedback="No feedback"),
                    summary=CritiqueSection(score=0, feedback="No feedback"),
                    personal_info=CritiqueSection(score=0, feedback="No feedback")
                ),
                suggestions=critique.suggestions or [],
                strengths=critique.strengths or [],
                weaknesses=critique.weaknesses or [],
                ai_model=critique.ai_model
            ),
            created_at=critique.created_at
        )
        for critique in critiques
    ]



@router.post("/{resume_id}/reparse")
async def reparse_resume(
    resume_id: int,
    background: BackgroundTasks,
    enhance_images: bool = True,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Endpoint to reparse a resume with different options."""
    profile = await ProfileCRUD.get_by_user_id(db, user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    resume = await ResumeCRUD.get_resume_by_id(db, resume_id, profile.id)
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Read file content
    fs = FileService()
    content = await fs.read_file(resume.file_path)
    
    # Add background task for comprehensive re-parsing
    async def reparse_background(
        resume_id: int, 
        profile_id: int, 
        file_content: bytes, 
        filename: str, 
        enhance_images: bool
    ):
        """Background task for comprehensive re-parsing."""
        async for db_session in db_manager.get_session():
            try:
                await ResumeCRUD.update_resume(
                    db_session, 
                    resume_id, 
                    {"parse_status": ParseStatus.PROCESSING}
                )
                await db_session.commit()

                parser = ResumeParserService()
                parsed_data = await parser.parse(file_content, filename, enhance_images=enhance_images)

                await ResumeCRUD.update_resume(db_session, resume_id, {
                    "parse_status": ParseStatus.COMPLETED,
                    "parsed_data": parsed_data.model_dump(),
                    "confidence_score": parsed_data.confidence_score or 0.85
                })
                await db_session.commit()

                # *** NEW: Process and store comprehensive resume data ***
                resume_service = ResumeDataService()
                await resume_service.process_parsed_resume_data(
                    db_session, profile_id,resume_id, parsed_data
                )
                
                print(f"Successfully reparsed and processed resume {resume_id}")
                
            except Exception as e:
                await db_session.rollback()
                await ResumeCRUD.update_resume(
                    db_session, 
                    resume_id, 
                    {"parse_status": ParseStatus.FAILED}
                )
                await db_session.commit()
                print(f"Reparsing failed for resume {resume_id}: {e}")
            break
    
    background.add_task(reparse_background, resume.id, profile.id, content, resume.original_name, enhance_images)
    
    return {"message": "Reparsing started in background"}


# New endpoint to get comprehensive profile data
@router.get("/profile/complete-data")
async def get_complete_profile_data(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get complete profile data including work experience, education, and skills."""
    from app.crud.resume_crud import WorkExperienceCRUD, EducationCRUD, ProfileSkillCRUD
    
    profile = await ProfileCRUD.get_by_user_id(db, user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Get all related data
    work_experiences = await WorkExperienceCRUD.get_by_profile_id(db, profile.id)
    educations = await EducationCRUD.get_by_profile_id(db, profile.id)
    skills = await ProfileSkillCRUD.get_by_profile_id(db, profile.id)
    
    return {
        "profile": {
            "id": profile.id,
            "first_name": profile.first_name,
            "last_name": profile.last_name,
            "phone": profile.phone,
            "location": profile.location,
            "bio": profile.bio,
            "linkedin_url": profile.linkedin_url,
            "github_url": profile.github_url,
            "website_url": profile.website_url,
            "profile_completeness": profile.profile_completeness
        },
        "work_experiences": [
            {
                "id": exp.id,
                "company": exp.company,
                "position": exp.position,
                "location": exp.location,
                "start_date": exp.start_date.isoformat() if exp.start_date else None,
                "end_date": exp.end_date.isoformat() if exp.end_date else None,
                "is_current": exp.is_current,
                "description": exp.description,
                "achievements": exp.achievements,
                "skills": exp.skills
            }
            for exp in work_experiences
        ],
        "educations": [
            {
                "id": edu.id,
                "institution": edu.institution,
                "degree": edu.degree,
                "field": edu.field,
                "start_date": edu.start_date.isoformat() if edu.start_date else None,
                "end_date": edu.end_date.isoformat() if edu.end_date else None,
                "grade": edu.grade,
                "description": edu.description
            }
            for edu in educations
        ],
        "skills": [
            {
                "id": skill.id,
                "skill_name": skill.skill_name,
                "category": skill.category,
                "proficiency_level": skill.proficiency_level,
                "years_experience": skill.years_experience,
                "source": skill.source,
                "verified": skill.verified
            }
            for skill in skills
        ]
    }
