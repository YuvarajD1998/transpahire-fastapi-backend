from fastapi import HTTPException, UploadFile
from app.config import settings

def validate_upload(file: UploadFile) -> None:
    if file.content_type not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")
