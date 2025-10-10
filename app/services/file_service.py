"""
Unstructured Resume Parser Service
Integrates unstructured library with existing FastAPI resume parsing pipeline
Supports: .docx, .pdf (text & scanned), .png, .jpg, .jpeg
Python 3.10+
"""

# Installation requirements:
# pip install "unstructured[all-docs]"
# pip install pillow
# 
# System dependencies (install via apt/brew/etc):
# - tesseract-ocr (for OCR on scanned PDFs and images)
# - poppler-utils (for PDF processing)
# - libmagic-dev (for file type detection)

import io
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Unstructured imports
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.image import partition_image

# Optional: Image preprocessing for better OCR
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# Your existing imports
from app.config import settings
from app.services.huggingface_service import HuggingFaceService
from app.services.openai_service import OpenAIService
from app.models.schemas import ParsedResumeData, ParsedSkill, ParsedExperience, ParsedEducation
from app.services.gemini_service import GeminiService

# Enhanced regular expressions for contact info extraction
EMAIL_RE = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_RE = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
LINKEDIN_RE = r'(https?://)?(www\.)?linkedin\.com/(in|pub)/[a-zA-Z0-9-]+'
GITHUB_RE = r'(https?://)?(www\.)?github\.com/[a-zA-Z0-9-]+'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnstructuredExtractor:
    """
    Core extractor using Unstructured library.
    Handles text, table, and image extraction with automatic OCR.
    """
    
    # Supported file formats
    SUPPORTED_FORMATS = {
        '.docx': 'word',
        '.doc': 'word',
        '.pdf': 'pdf',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.tiff': 'image',
        '.bmp': 'image'
    }
    
    def __init__(self, temp_dir: str = "temp_extraction"):
        """Initialize the extractor with temporary directory for processing."""
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_bytes: Image content as bytes
            
        Returns:
            Preprocessed image as bytes
        """
        if not PILLOW_AVAILABLE:
            logger.warning("Pillow not available, skipping image preprocessing")
            return image_bytes
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Apply median filter to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Save to bytes
            output = io.BytesIO()
            image.save(output, format='PNG')
            output.seek(0)
            
            logger.info("Image preprocessing completed")
            return output.read()
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image_bytes
    
    async def extract_with_unstructured(
        self, 
        file_content: bytes, 
        filename: str,
        enhance_images: bool = True,
        use_hi_res: bool = True
    ) -> Dict[str, Any]:
        """
        Extract structured data using Unstructured library.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            enhance_images: Apply image preprocessing
            use_hi_res: Use high-resolution strategy for better table extraction
            
        Returns:
            Structured data with headings, paragraphs, tables, images
        """
        file_ext = self._get_file_extension(filename)
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        file_type = self.SUPPORTED_FORMATS[file_ext]
        logger.info(f"Processing {file_type} file: {filename}")
        
        try:
            # Create temporary file for processing
            temp_file = self.temp_dir / f"temp_{filename}"
            
            # Preprocess images if requested
            processed_content = file_content
            if file_type == 'image' and enhance_images:
                processed_content = self.preprocess_image(file_content)
            
            # Write to temporary file (unstructured requires file path)
            with open(temp_file, 'wb') as f:
                f.write(processed_content)
            
            # Configure extraction parameters
            extraction_kwargs = {
                'filename': str(temp_file),
                'include_page_breaks': True,
            }
            
            # For PDFs and images, use hi_res strategy for better table extraction
            if file_type in ['pdf', 'image'] and use_hi_res:
                extraction_kwargs['strategy'] = 'hi_res'
                extraction_kwargs['infer_table_structure'] = True
            
            # Use specific partition function based on file type
            if file_type == 'pdf':
                elements = partition_pdf(**extraction_kwargs)
            elif file_type == 'word':
                elements = partition_docx(**extraction_kwargs)
            elif file_type == 'image':
                elements = partition_image(**extraction_kwargs)
            else:
                # Fallback to auto partition
                elements = partition(**extraction_kwargs)
            
            logger.info(f"Extracted {len(elements)} elements from {filename}")
            
            # Process elements into structured format
            structured_data = self._process_elements(elements, filename)
            
            # Clean up temporary file
            try:
                temp_file.unlink()
            except:
                pass
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Unstructured extraction failed for {filename}: {e}")
            # Clean up on error
            try:
                temp_file.unlink()
            except:
                pass
            raise RuntimeError(f"Failed to extract with unstructured: {str(e)}")
    
    def _process_elements(self, elements: List[Any], filename: str) -> Dict[str, Any]:
        """
        Process extracted elements into structured format.
        
        Args:
            elements: List of elements from Unstructured
            filename: Original filename
            
        Returns:
            Structured data dictionary
        """
        categorized = {
            'titles': [],
            'headings': [],
            'paragraphs': [],
            'tables': [],
            'lists': [],
            'images': [],
            'other': []
        }
        
        # Categorize elements
        for element in elements:
            element_type = element.category if hasattr(element, 'category') else 'unknown'
            
            if element_type == 'Title':
                categorized['titles'].append(element)
            elif element_type in ['Header', 'Heading']:
                categorized['headings'].append(element)
            elif element_type in ['NarrativeText', 'Text']:
                categorized['paragraphs'].append(element)
            elif element_type == 'Table':
                categorized['tables'].append(element)
            elif element_type in ['ListItem', 'BulletedText']:
                categorized['lists'].append(element)
            elif element_type == 'Image':
                categorized['images'].append(element)
            else:
                categorized['other'].append(element)
        
        # Build structured output
        structured_data = {
            "headings": [],
            "paragraphs": [],
            "tables": [],
            "images": [],
            "raw_text": "",
            "metadata": {
                "filename": filename,
                "parsed_at": datetime.now().isoformat(),
                "total_elements": len(elements)
            }
        }
        
        # Extract headings
        for element in categorized['titles'] + categorized['headings']:
            text = self._clean_text(str(element))
            if text:
                structured_data['headings'].append(text)
        
        # Extract paragraphs (including lists and other text)
        all_text_parts = []
        for element in categorized['paragraphs'] + categorized['lists'] + categorized['other']:
            text = self._clean_text(str(element))
            if text:
                structured_data['paragraphs'].append(text)
                all_text_parts.append(text)
        
        # Extract tables
        for table_element in categorized['tables']:
            table_data = self._extract_table_data(table_element)
            if table_data:
                structured_data['tables'].append(table_data)
                # Add table text to raw_text
                table_text = self._table_to_text(table_data)
                all_text_parts.append(table_text)
        
        # Extract images
        for idx, img_element in enumerate(categorized['images']):
            structured_data['images'].append(f"image_{idx}")
        
        # Combine all text for LLM processing
        structured_data['raw_text'] = "\n\n".join(
            structured_data['headings'] + all_text_parts
        )
        
        logger.info(f"Processed - Headings: {len(structured_data['headings'])}, "
                   f"Paragraphs: {len(structured_data['paragraphs'])}, "
                   f"Tables: {len(structured_data['tables'])}")
        
        return structured_data
    
    def _extract_table_data(self, table_element: Any) -> List[List[str]]:
        """Extract table data preserving row and cell structure."""
        try:
            # Try to get HTML representation first (most structured)
            if hasattr(table_element, 'metadata') and hasattr(table_element.metadata, 'text_as_html'):
                html_text = table_element.metadata.text_as_html
                if html_text:
                    rows = self._parse_html_table(html_text)
                    if rows:
                        return rows
            
            # Fallback: parse text representation
            table_text = str(table_element)
            rows = []
            for line in table_text.split('\n'):
                line = line.strip()
                if line:
                    cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                    if not cells:
                        cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
                    if cells:
                        rows.append(cells)
            
            return rows if rows else [[table_text]]
            
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            return [[str(table_element)]]
    
    def _parse_html_table(self, html: str) -> List[List[str]]:
        """Basic HTML table parser."""
        try:
            # Extract rows
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            rows = re.findall(row_pattern, html, re.DOTALL | re.IGNORECASE)
            
            table_data = []
            for row in rows:
                # Extract cells (th or td)
                cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
                cells = re.findall(cell_pattern, row, re.DOTALL | re.IGNORECASE)
                
                # Clean HTML tags from cell content
                clean_cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]
                if clean_cells:
                    table_data.append(clean_cells)
            
            return table_data
            
        except Exception as e:
            logger.warning(f"HTML table parsing failed: {e}")
            return []
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text format."""
        try:
            return "\n".join([" | ".join(row) for row in table_data])
        except:
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        text = ' '.join(text.split())
        return text.strip()
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        if '.' in filename:
            return '.' + filename.split('.')[-1].lower()
        return ''


class ResumeParserService:
    """
    Enhanced Resume Parser Service using Unstructured library.
    Integrates with existing Gemini/OpenAI/HuggingFace parsers.
    """
    
    def __init__(self) -> None:
        self.gemini = GeminiService()      # Primary parser
        self.openai = OpenAIService()      # First fallback
        self.hf = HuggingFaceService()     # Second fallback
        self.unstructured = UnstructuredExtractor()  # NEW: Unstructured extractor
        logger.info("ResumeParserService initialized with Unstructured support")

    async def parse(
        self, 
        file_content: bytes, 
        filename: str, 
        enhance_images: bool = True
    ) -> ParsedResumeData:
        """
        Enhanced parsing with Unstructured library for extraction.
        Uses Gemini as primary parser, OpenAI and HuggingFace as fallbacks.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            enhance_images: Apply image preprocessing for better OCR
            
        Returns:
            ParsedResumeData object
        """
        try:
            # STEP 1: Extract text using Unstructured library
            logger.info(f"Extracting text from {filename} using Unstructured library")
            
            try:
                # Try unstructured extraction first
                structured_data = await self.unstructured.extract_with_unstructured(
                    file_content, 
                    filename, 
                    enhance_images=enhance_images,
                    use_hi_res=True  # Enable high-res for better tables
                )
                
                # Get the raw text for LLM processing
                text = structured_data.get('raw_text', '')
                
                # Store structured data for potential use
                self._structured_cache = structured_data
                
                logger.info(f"Unstructured extraction successful: {len(text)} chars extracted")
                
            except Exception as unstructured_error:
                logger.warning(f"Unstructured extraction failed: {unstructured_error}, using fallback")
                # Fallback to legacy extraction
                text = await self._legacy_extract_text(file_content, filename, enhance_images)
                self._structured_cache = None
            
            if not text.strip():
                raise ValueError("No text extracted from file")
            
            # STEP 2: Parse extracted text with LLM services
            # Try Gemini first (best accuracy and multimodal capabilities)
            if self.gemini.is_available():
                try:
                    logger.info("Attempting to parse with Gemini...")
                    return await self.gemini.parse_resume_text(text)
                except Exception as gemini_error:
                    logger.warning(f"Gemini parsing failed: {gemini_error}, falling back to OpenAI")
            
            # Fallback to OpenAI
            try:
                logger.info("Attempting to parse with OpenAI...")
                return await self.openai.parse_resume_text(text)
            except Exception as openai_error:
                logger.warning(f"OpenAI parsing failed: {openai_error}, falling back to HuggingFace")
                
                # Fallback to HuggingFace
                try:
                    logger.info("Attempting to parse with HuggingFace...")
                    return await self.hf.parse_resume_text(text)
                except Exception as hf_error:
                    logger.warning(f"HuggingFace parsing failed: {hf_error}, falling back to enhanced regex")
                    
                    # Final fallback to enhanced regex parsing
                    return await self._enhanced_fallback_parse(text)
                    
        except Exception as e:
            logger.error(f"All parsing strategies failed: {e}")
            # Return minimal structured data
            return ParsedResumeData(
                personal_info={},
                skills=[],
                experience=[],
                education=[],
                summary=None,
                confidence_score=0.1
            )
    
    async def parse_file(
        self, 
        file_content: bytes, 
        filename: str, 
        enhance_images: bool = True
    ) -> ParsedResumeData:
        """Alias for parse method to maintain compatibility."""
        return await self.parse(file_content, filename, enhance_images)
    
    async def extract_text(
        self, 
        file_content: bytes, 
        filename: str, 
        enhance_images: bool = True
    ) -> str:
        """
        Public method to extract text from any supported resume file.
        Uses Unstructured library as primary method.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            enhance_images: Apply image preprocessing
            
        Returns:
            Extracted text
        """
        try:
            # Try unstructured extraction
            structured_data = await self.unstructured.extract_with_unstructured(
                file_content, 
                filename, 
                enhance_images=enhance_images,
                use_hi_res=True
            )
            return structured_data.get('raw_text', '')
            
        except Exception as e:
            logger.warning(f"Unstructured extraction failed: {e}, using legacy method")
            # Fallback to legacy extraction
            return await self._legacy_extract_text(file_content, filename, enhance_images)
    
    async def _legacy_extract_text(
        self, 
        file_content: bytes, 
        filename: str, 
        enhance_images: bool = True
    ) -> str:
        """
        Legacy extraction method (fallback).
        Uses PyPDF2, docx, pytesseract for extraction.
        """
        import PyPDF2
        import docx
        import pytesseract
        from pdf2image import convert_from_bytes
        
        file_extension = self._get_file_extension(filename)
        
        try:
            if file_extension in ['.pdf']:
                return await self._extract_from_pdf_legacy(file_content, enhance_images)
            elif file_extension in ['.docx', '.doc']:
                return await self._extract_from_docx_legacy(file_content)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return await self._extract_from_image_legacy(file_content, enhance_images)
            else:
                # Try to decode as plain text
                return file_content.decode(errors="ignore")
        except Exception as e:
            logger.error(f"Legacy extraction failed: {e}")
            return ""
    
    async def _extract_from_pdf_legacy(self, data: bytes, enhance_images: bool = True) -> str:
        """Legacy PDF extraction using PyPDF2."""
        import PyPDF2
        from pdf2image import convert_from_bytes
        import pytesseract
        
        try:
            buf = io.BytesIO(data)
            reader = PyPDF2.PdfReader(buf)
            
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            extracted_text = "\n".join(text_parts).strip()
            
            # If little text, try OCR
            if len(extracted_text) < 100:
                logger.info("PDF appears to be image-based, using OCR...")
                images = convert_from_bytes(data)
                ocr_parts = []
                for i, image in enumerate(images):
                    if enhance_images:
                        image = self._preprocess_image_pil(image)
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    if page_text.strip():
                        ocr_parts.append(page_text)
                return "\n\n".join(ocr_parts).strip() if ocr_parts else extracted_text
            
            return extracted_text
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    async def _extract_from_docx_legacy(self, data: bytes) -> str:
        """Legacy DOCX extraction."""
        import docx
        
        try:
            buf = io.BytesIO(data)
            doc = docx.Document(buf)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            return "\n".join(text_parts).strip()
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
    
    async def _extract_from_image_legacy(self, data: bytes, enhance_images: bool = True) -> str:
        """Legacy image extraction using pytesseract."""
        import pytesseract
        from PIL import Image
        
        try:
            image = Image.open(io.BytesIO(data))
            
            if enhance_images:
                image = self._preprocess_image_pil(image)
            
            text = pytesseract.image_to_string(image, lang='eng')
            return text.strip()
        except Exception as e:
            raise Exception(f"Image extraction failed: {str(e)}")
    
    def _preprocess_image_pil(self, image):
        """Preprocess PIL image for OCR."""
        if image.mode != 'L':
            image = image.convert('L')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    
    async def _enhanced_fallback_parse(self, text: str) -> ParsedResumeData:
        """Enhanced fallback parsing using regex."""
        personal_info = self._extract_personal_info(text)
        skills = await self._extract_skills_enhanced(text)
        experience = self._extract_experience_enhanced(text)
        education = self._extract_education_enhanced(text)
        summary = self._extract_summary(text)
        
        return ParsedResumeData(
            personal_info=personal_info,
            skills=skills,
            experience=experience,
            education=education,
            summary=summary,
            confidence_score=0.6
        )
    
    def _extract_personal_info(self, text: str) -> Dict[str, Any]:
        """Extract personal information."""
        email_match = re.search(EMAIL_RE, text, re.IGNORECASE)
        phone_match = re.search(PHONE_RE, text)
        linkedin_match = re.search(LINKEDIN_RE, text, re.IGNORECASE)
        github_match = re.search(GITHUB_RE, text, re.IGNORECASE)
        name = self._extract_name(text)
        
        return {
            "name": name,
            "email": email_match.group(0) if email_match else None,
            "phone": phone_match.group(0) if phone_match else None,
            "linkedin": linkedin_match.group(0) if linkedin_match else None,
            "github": github_match.group(0) if github_match else None,
        }
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name from text."""
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(word.istitle() or word.isupper() for word in words if len(word) > 1):
                    if not any(char in line for char in ['@', 'http', '://', '•', '·']):
                        return line
        return None
    
    async def _extract_skills_enhanced(self, text: str) -> List[ParsedSkill]:
        """Extract skills using comprehensive database."""
        skill_categories = {
            'Programming': ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust'],
            'Web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
            'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis'],
            'Cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'Data Science': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'machine learning'],
        }
        
        found_skills = []
        text_lower = text.lower()
        
        for category, skill_list in skill_categories.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    level = "INTERMEDIATE"
                    if any(pro in text_lower for pro in ['expert', 'advanced', 'senior']):
                        level = "EXPERT"
                    
                    found_skills.append(ParsedSkill(
                        name=skill.title(),
                        category=category,
                        proficiency_level=level
                    ))
        
        return found_skills
    
    def _extract_experience_enhanced(self, text: str) -> List[ParsedExperience]:
        """Extract experience using pattern matching."""
        experiences = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if any(keyword in line.lower() for keyword in ['inc', 'corp', 'ltd', 'company', 'technologies']):
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if any(role in next_line.lower() for role in ['engineer', 'developer', 'manager']):
                        experiences.append(ParsedExperience(
                            company=line,
                            position=next_line,
                            description="Extracted from resume",
                            start_date=None,
                            end_date=None
                        ))
                        i += 2
                        continue
            i += 1
        
        return experiences
    
    def _extract_education_enhanced(self, text: str) -> List[ParsedEducation]:
        """Extract education."""
        education = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(edu in line_lower for edu in ['university', 'college', 'bachelor', 'master']):
                education.append(ParsedEducation(
                    institution=line.strip(),
                    degree="Not specified",
                    field=None
                ))
        
        return education
    
    def _extract_summary(self, text: str) -> Optional[str]:
        """Extract summary."""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if 50 < len(line) < 300:
                if not any(kw in line.lower() for kw in ['email', 'phone', 'linkedin']):
                    return line
        return None
    
    def validate_file_type(self, filename: str) -> bool:
        """Validate if file type is supported."""
        supported = ['.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        return self._get_file_extension(filename) in supported
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension."""
        if '.' in filename:
            return '.' + filename.split('.')[-1].lower()
        return ''


# Optional: Keep FileService if needed for other purposes
class FileService:
    """Service for handling file operations."""
    
    def __init__(self, upload_dir: str = "uploads"):
        import uuid
        import aiofiles
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_file(self, content: bytes, filename: str, content_type: str) -> str:
        """Save uploaded file to disk."""
        import uuid
        import aiofiles
        
        file_id = str(uuid.uuid4())
        file_extension = self._get_file_extension(filename)
        new_filename = f"{file_id}{file_extension}"
        file_path = self.upload_dir / new_filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        return str(file_path)
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from disk."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except OSError:
            return False
    
    async def read_file(self, file_path: str) -> bytes:
        """Read file content."""
        import aiofiles
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension."""
        if '.' in filename:
            return '.' + filename.split('.')[-1].lower()
        return ''
