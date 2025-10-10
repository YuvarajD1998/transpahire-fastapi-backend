"""
Resume Parser using Unstructured Library
Supports: .docx, .pdf (text & scanned), .png, .jpg, .jpeg
Extracts: Text, tables, images with automatic OCR
Output: Structured JSON ready for NLP/LLM processing
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
#
# Optional for better OCR:
# pip install unstructured-inference
# pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Unstructured imports
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.image import partition_image
from unstructured.staging.base import elements_to_json

# Optional: Image preprocessing for better OCR
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnstructuredResumeParser:
    """
    Enhanced resume parser using Unstructured library.
    Extracts text, tables, and images from resumes with automatic OCR.
    """
    
    # Supported file formats
    SUPPORTED_FORMATS = {
        '.docx': 'word',
        '.pdf': 'pdf',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image'
    }
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the resume parser.
        
        Args:
            output_dir: Directory to save extracted images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized UnstructuredResumeParser with output_dir: {self.output_dir}")
    
    def preprocess_image(self, image_path: str) -> Optional[str]:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to preprocessed image or None if preprocessing failed
        """
        if not PILLOW_AVAILABLE:
            logger.warning("Pillow not available, skipping image preprocessing")
            return image_path
        
        try:
            image = Image.open(image_path)
            
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
            
            # Save preprocessed image
            preprocessed_path = str(self.output_dir / f"preprocessed_{Path(image_path).name}")
            image.save(preprocessed_path)
            
            logger.info(f"Preprocessed image saved to: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image_path
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if the file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if supported, False otherwise
        """
        file_ext = Path(file_path).suffix.lower()
        is_valid = file_ext in self.SUPPORTED_FORMATS
        
        if not is_valid:
            logger.error(f"Unsupported file format: {file_ext}")
        
        return is_valid
    
    def extract_elements(
        self, 
        file_path: str, 
        enhance_images: bool = True,
        use_hi_res: bool = True
    ) -> List[Any]:
        """
        Extract elements from file using Unstructured library.
        
        Args:
            file_path: Path to the resume file
            enhance_images: Apply image preprocessing for better OCR
            use_hi_res: Use high-resolution strategy for better table extraction
            
        Returns:
            List of extracted elements
        """
        if not self.validate_file(file_path):
            raise ValueError(f"Unsupported file format: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        file_type = self.SUPPORTED_FORMATS[file_ext]
        
        logger.info(f"Processing {file_type} file: {file_path}")
        
        try:
            # Preprocess images if requested
            processed_path = file_path
            if file_type == 'image' and enhance_images:
                processed_path = self.preprocess_image(file_path) or file_path
            
            # Configure extraction parameters
            extraction_kwargs = {
                'filename': processed_path,
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
            
            logger.info(f"Extracted {len(elements)} elements from {file_path}")
            return elements
            
        except Exception as e:
            logger.error(f"Extraction failed for {file_path}: {e}")
            raise RuntimeError(f"Failed to extract elements from {file_path}: {str(e)}")
    
    def categorize_elements(self, elements: List[Any]) -> Dict[str, List[Any]]:
        """
        Categorize extracted elements by type.
        
        Args:
            elements: List of elements from Unstructured
            
        Returns:
            Dictionary with categorized elements
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
        
        logger.info(f"Categorized elements: {[(k, len(v)) for k, v in categorized.items()]}")
        return categorized
    
    def extract_table_data(self, table_element: Any) -> List[List[str]]:
        """
        Extract table data preserving row and cell structure.
        
        Args:
            table_element: Table element from Unstructured
            
        Returns:
            Table as list of rows, rows as lists of cells
        """
        try:
            # Try to get HTML representation first (most structured)
            if hasattr(table_element, 'metadata') and hasattr(table_element.metadata, 'text_as_html'):
                html_text = table_element.metadata.text_as_html
                if html_text:
                    # Parse HTML table (basic parsing)
                    rows = self._parse_html_table(html_text)
                    if rows:
                        return rows
            
            # Fallback: parse text representation
            table_text = str(table_element)
            rows = []
            for line in table_text.split('\n'):
                line = line.strip()
                if line:
                    # Simple split by whitespace/tabs
                    cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                    if not cells:
                        cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
                    if cells:
                        rows.append(cells)
            
            return rows if rows else [[table_text]]
            
        except Exception as e:
            logger.warning(f"Table extraction failed, using raw text: {e}")
            return [[str(table_element)]]
    
    def _parse_html_table(self, html: str) -> List[List[str]]:
        """
        Basic HTML table parser.
        
        Args:
            html: HTML string containing table
            
        Returns:
            Table as list of rows
        """
        try:
            import re
            
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
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and empty strings.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def parse_resume(
        self, 
        file_path: str, 
        enhance_images: bool = True,
        use_hi_res: bool = True
    ) -> Dict[str, Any]:
        """
        Main parsing function that extracts structured data from resume.
        
        Args:
            file_path: Path to resume file
            enhance_images: Apply image preprocessing
            use_hi_res: Use high-resolution extraction strategy
            
        Returns:
            Structured JSON object with resume data
        """
        logger.info(f"Starting resume parsing: {file_path}")
        
        try:
            # Extract elements
            elements = self.extract_elements(file_path, enhance_images, use_hi_res)
            
            # Categorize elements
            categorized = self.categorize_elements(elements)
            
            # Build structured output
            structured_data = {
                "headings": [],
                "paragraphs": [],
                "tables": [],
                "images": [],
                "metadata": {
                    "filename": Path(file_path).name,
                    "file_path": str(file_path),
                    "parsed_at": datetime.now().isoformat(),
                    "total_elements": len(elements)
                }
            }
            
            # Extract headings (titles + headings)
            for element in categorized['titles'] + categorized['headings']:
                text = self.clean_text(str(element))
                if text:
                    structured_data['headings'].append(text)
            
            # Extract paragraphs (narrative text + lists + other)
            for element in categorized['paragraphs'] + categorized['lists'] + categorized['other']:
                text = self.clean_text(str(element))
                if text:
                    structured_data['paragraphs'].append(text)
            
            # Extract tables
            for table_element in categorized['tables']:
                table_data = self.extract_table_data(table_element)
                if table_data:
                    structured_data['tables'].append(table_data)
            
            # Extract images (placeholders or file references)
            for idx, img_element in enumerate(categorized['images']):
                image_info = {
                    "index": idx,
                    "type": "image",
                    "metadata": {}
                }
                
                # Try to get image metadata
                if hasattr(img_element, 'metadata'):
                    if hasattr(img_element.metadata, 'image_path'):
                        image_info['path'] = img_element.metadata.image_path
                    if hasattr(img_element.metadata, 'page_number'):
                        image_info['metadata']['page'] = img_element.metadata.page_number
                
                # Placeholder if no path
                if 'path' not in image_info:
                    image_info['path'] = f"image_{idx}.png"
                
                structured_data['images'].append(image_info['path'])
            
            logger.info(f"Successfully parsed resume: {file_path}")
            logger.info(f"Extracted - Headings: {len(structured_data['headings'])}, "
                       f"Paragraphs: {len(structured_data['paragraphs'])}, "
                       f"Tables: {len(structured_data['tables'])}, "
                       f"Images: {len(structured_data['images'])}")
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Resume parsing failed for {file_path}: {e}")
            raise RuntimeError(f"Failed to parse resume: {str(e)}")
    
    def parse_resume_to_json_file(
        self, 
        file_path: str, 
        output_path: Optional[str] = None,
        enhance_images: bool = True,
        use_hi_res: bool = True
    ) -> str:
        """
        Parse resume and save to JSON file.
        
        Args:
            file_path: Path to resume file
            output_path: Path to save JSON (optional)
            enhance_images: Apply image preprocessing
            use_hi_res: Use high-resolution extraction
            
        Returns:
            Path to saved JSON file
        """
        structured_data = self.parse_resume(file_path, enhance_images, use_hi_res)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = Path(file_path).stem
            output_path = self.output_dir / f"{base_name}_parsed.json"
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved parsed data to: {output_path}")
        return str(output_path)
    
    def batch_parse_resumes(
        self, 
        input_dir: str, 
        output_dir: Optional[str] = None,
        enhance_images: bool = True,
        use_hi_res: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple resumes in a folder.
        
        Args:
            input_dir: Directory containing resume files
            output_dir: Directory to save parsed JSON files
            enhance_images: Apply image preprocessing
            use_hi_res: Use high-resolution extraction
            
        Returns:
            Summary of batch processing
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all supported files
        resume_files = []
        for ext in self.SUPPORTED_FORMATS.keys():
            resume_files.extend(input_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(resume_files)} resume files in {input_dir}")
        
        # Process each file
        results = {
            "total": len(resume_files),
            "successful": 0,
            "failed": 0,
            "files": []
        }
        
        for resume_file in resume_files:
            try:
                output_path = output_dir / f"{resume_file.stem}_parsed.json"
                self.parse_resume_to_json_file(
                    str(resume_file), 
                    str(output_path),
                    enhance_images=enhance_images,
                    use_hi_res=use_hi_res
                )
                
                results['successful'] += 1
                results['files'].append({
                    "filename": resume_file.name,
                    "status": "success",
                    "output": str(output_path)
                })
                
            except Exception as e:
                logger.error(f"Failed to process {resume_file.name}: {e}")
                results['failed'] += 1
                results['files'].append({
                    "filename": resume_file.name,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Save batch summary
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing complete. Summary saved to: {summary_path}")
        logger.info(f"Success: {results['successful']}, Failed: {results['failed']}")
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_single_file():
    """Example: Parse a single resume file"""
    parser = UnstructuredResumeParser(output_dir="parsed_resumes")
    
    # Parse a single file
    resume_path = "sample_resume.pdf"  # Change to your file
    
    try:
        # Get structured data
        parsed_data = parser.parse_resume(
            resume_path, 
            enhance_images=True,  # Enable image preprocessing
            use_hi_res=True       # Enable high-res for better tables
        )
        
        # Print results
        print(json.dumps(parsed_data, indent=2))
        
        # Save to JSON file
        output_file = parser.parse_resume_to_json_file(resume_path)
        print(f"\nParsed data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example: Process multiple resumes in a folder"""
    parser = UnstructuredResumeParser(output_dir="parsed_resumes")
    
    # Process all resumes in a directory
    results = parser.batch_parse_resumes(
        input_dir="resumes_folder",      # Folder containing resumes
        output_dir="parsed_resumes",     # Where to save JSON files
        enhance_images=True,
        use_hi_res=True
    )
    
    # Print summary
    print(f"\nBatch Processing Summary:")
    print(f"Total files: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    
    print("\nDetailed results:")
    for file_info in results['files']:
        status = file_info['status']
        filename = file_info['filename']
        if status == 'success':
            print(f"✓ {filename} -> {file_info['output']}")
        else:
            print(f"✗ {filename} -> Error: {file_info['error']}")


def example_nlp_ready_output():
    """Example: Parse and prepare for NLP/LLM processing"""
    parser = UnstructuredResumeParser(output_dir="parsed_resumes")
    
    resume_path = "sample_resume.pdf"
    
    try:
        parsed_data = parser.parse_resume(resume_path)
        
        # Prepare for NLP/LLM - combine all text
        full_text = "\n\n".join([
            "HEADINGS:",
            "\n".join(parsed_data['headings']),
            "\nCONTENT:",
            "\n".join(parsed_data['paragraphs'])
        ])
        
        # This clean text can now be passed to your NLP/LLM pipeline
        print("=== NLP-Ready Text ===")
        print(full_text[:500] + "...")  # Print first 500 chars
        
        # Example: Pass to your existing NLP parser
        # parsed_fields = your_llm_parser.parse(full_text)
        # Extract: name, email, skills, education, experience, etc.
        
        print("\n=== Structured Data for LLM ===")
        print(f"Headings extracted: {len(parsed_data['headings'])}")
        print(f"Paragraphs extracted: {len(parsed_data['paragraphs'])}")
        print(f"Tables extracted: {len(parsed_data['tables'])}")
        
        # Tables can be passed separately for structured extraction
        if parsed_data['tables']:
            print("\nFirst table:")
            print(json.dumps(parsed_data['tables'][0], indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Resume Parser using Unstructured Library\n")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # Example 1: Parse single file
    # example_single_file()
    
    # Example 2: Batch process multiple files
    # example_batch_processing()
    
    # Example 3: Prepare output for NLP/LLM
    # example_nlp_ready_output()
    
    print("\nExamples available:")
    print("1. example_single_file() - Parse one resume")
    print("2. example_batch_processing() - Parse multiple resumes")
    print("3. example_nlp_ready_output() - Prepare for NLP/LLM processing")
