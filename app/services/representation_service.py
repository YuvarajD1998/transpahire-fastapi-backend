import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResumeRepresentationService:
    """
    Service to generate focused text representations for embeddings.
    Converts structured resume data into optimized strings for semantic search.
    """
    
    @staticmethod
    def generate_summary_text(profile_data: Dict) -> str:
        """
        Generate a concise professional summary for high-level matching.
        
        Example output:
        "Senior Frontend Developer with 3.5 years experience. 
         Key strengths: React, Redux, JavaScript, TypeScript, API integrations.
         Worked in Fintech domain at Razorpay."
        """
        parts = []
        
        profile = profile_data.get("profile", {})
        work_experiences = profile_data.get("work_experiences", [])
        skills = profile_data.get("skills", [])
        
        # Title and experience duration
        if work_experiences:
            latest_position = work_experiences[0].position if hasattr(work_experiences[0], 'position') else None
            total_years = ResumeRepresentationService._calculate_total_experience(work_experiences)
            
            if latest_position and total_years:
                parts.append(f"{latest_position} with {total_years} years experience")
        
        # Key skills (top 5-7)
        if skills:
            top_skills = [skill.skill_name if hasattr(skill, 'skill_name') else skill['skill_name'] 
                         for skill in skills[:7]]
            if top_skills:
                parts.append(f"Key strengths: {', '.join(top_skills)}")
        
        # Domain/Industry context
        if work_experiences:
            recent_companies = [exp.company if hasattr(exp, 'company') else exp['company'] 
                              for exp in work_experiences[:2]]
            if recent_companies:
                parts.append(f"Worked at {', '.join(recent_companies)}")
        
        # Professional summary from bio
        if hasattr(profile, 'bio') and profile.bio:
            parts.append(profile.bio[:200])
        
        return ". ".join(parts)
    
    @staticmethod
    def generate_skills_text(profile_data: Dict) -> str:
        """
        Generate skill-focused text with proficiency and years.
        
        Example output:
        "React (Advanced, 3 yrs), Redux (Intermediate, 2 yrs), 
         JavaScript (Advanced, 4 yrs), REST APIs, HTML CSS"
        """
        skills = profile_data.get("skills", [])
        
        if not skills:
            return ""
        
        skill_strings = []
        for skill in skills:
            skill_name = skill.skill_name if hasattr(skill, 'skill_name') else skill.get('skill_name', '')
            
            # Add proficiency if available
            proficiency = None
            if hasattr(skill, 'proficiency_level'):
                proficiency = skill.proficiency_level
            elif 'proficiency_level' in skill:
                proficiency = skill['proficiency_level']
            
            # Add years if available
            years = None
            if hasattr(skill, 'years_experience'):
                years = skill.years_experience
            elif 'years_experience' in skill:
                years = skill['years_experience']
            
            # Build skill string
            skill_str = skill_name
            if proficiency or years:
                details = []
                if proficiency:
                    details.append(str(proficiency))
                if years:
                    details.append(f"{years} yrs")
                skill_str += f" ({', '.join(details)})"
            
            skill_strings.append(skill_str)
        
        return ", ".join(skill_strings)
    
    @staticmethod
    def generate_experience_text(profile_data: Dict) -> str:
        """
        Generate experience-focused text with roles and responsibilities.
        
        Example output:
        "Company: Razorpay (Fintech). Role: Frontend Developer (React). 
         Built dashboards, onboarding flows, payment UI. Skills used: React, JS, Redux, APIs."
        """
        work_experiences = profile_data.get("work_experiences", [])
        
        if not work_experiences:
            return ""
        
        exp_strings = []
        for exp in work_experiences[:3]:  # Top 3 experiences
            company = exp.company if hasattr(exp, 'company') else exp.get('company', '')
            position = exp.position if hasattr(exp, 'position') else exp.get('position', '')
            description = exp.description if hasattr(exp, 'description') else exp.get('description', '')
            
            exp_str = f"Company: {company}. Role: {position}"
            
            # Add description (truncated)
            if description:
                desc_short = description[:200] + "..." if len(description) > 200 else description
                exp_str += f". {desc_short}"
            
            # Add skills used
            if hasattr(exp, 'skills') and exp.skills:
                skills_list = exp.skills if isinstance(exp.skills, list) else []
                if skills_list:
                    exp_str += f". Skills used: {', '.join(skills_list[:5])}"
            
            exp_strings.append(exp_str)
        
        return " | ".join(exp_strings)
    
    @staticmethod
    def generate_education_text(profile_data: Dict) -> str:
        """
        Generate education-focused text.
        
        Example output:
        "B.Tech Computer Science, MIT (2016-2020). 
         M.S. Software Engineering, Stanford (2020-2022)"
        """
        educations = profile_data.get("educations", [])
        
        if not educations:
            return ""
        
        edu_strings = []
        for edu in educations:
            degree = edu.degree if hasattr(edu, 'degree') else edu.get('degree', '')
            field = edu.field if hasattr(edu, 'field') else edu.get('field', '')
            institution = edu.institution if hasattr(edu, 'institution') else edu.get('institution', '')
            
            edu_str = f"{degree}"
            if field:
                edu_str += f" {field}"
            edu_str += f", {institution}"
            
            # Add dates if available
            start_date = edu.start_date if hasattr(edu, 'start_date') else edu.get('start_date')
            end_date = edu.end_date if hasattr(edu, 'end_date') else edu.get('end_date')
            
            if start_date and end_date:
                start_year = start_date.year if isinstance(start_date, datetime) else start_date
                end_year = end_date.year if isinstance(end_date, datetime) else end_date
                edu_str += f" ({start_year}-{end_year})"
            
            edu_strings.append(edu_str)
        
        return ". ".join(edu_strings)
    
    @staticmethod
    def generate_full_text(profile_data: Dict) -> str:
        """
        Generate comprehensive full resume text (fallback).
        Combines all sections with noise reduction.
        """
        parts = [
            ResumeRepresentationService.generate_summary_text(profile_data),
            ResumeRepresentationService.generate_skills_text(profile_data),
            ResumeRepresentationService.generate_experience_text(profile_data),
            ResumeRepresentationService.generate_education_text(profile_data)
        ]
        
        # Filter empty parts
        parts = [p for p in parts if p.strip()]
        
        return " | ".join(parts)
    
    @staticmethod
    def _calculate_total_experience(work_experiences: List) -> float:
        """Calculate total years of experience from work history."""
        total_months = 0
        
        for exp in work_experiences:
            start_date = exp.start_date if hasattr(exp, 'start_date') else exp.get('start_date')
            end_date = exp.end_date if hasattr(exp, 'end_date') else exp.get('end_date')
            is_current = exp.is_current if hasattr(exp, 'is_current') else exp.get('is_current', False)
            
            if start_date:
                start = start_date if isinstance(start_date, datetime) else datetime.now()
                if is_current:
                    end = datetime.now()
                elif end_date:
                    end = end_date if isinstance(end_date, datetime) else datetime.now()
                else:
                    continue
                
                months = (end.year - start.year) * 12 + (end.month - start.month)
                total_months += months
        
        return round(total_months / 12, 1)


class JobRepresentationService:
    """
    Service to generate focused text representations for job descriptions.
    """
    
    @staticmethod
    def generate_jd_summary(job_data: Dict) -> str:
        """
        Generate high-level job summary.
        
        Example:
        "Frontend Developer (2-4 yrs). React, Redux, TypeScript required. 
         Bangalore, Hybrid. Fintech domain preferred."
        """
        parts = []
        
        # Title and experience requirement
        title = job_data.get('title', '')
        requirements = job_data.get('requirements', {})
        
        if title:
            parts.append(title)
        
        # Experience range
        if isinstance(requirements, dict):
            min_exp = requirements.get('min_experience')
            max_exp = requirements.get('max_experience')
            if min_exp and max_exp:
                parts.append(f"({min_exp}-{max_exp} yrs)")
        
        # Key skills from requirements
        required_skills = JobRepresentationService._extract_required_skills(job_data)
        if required_skills:
            parts.append(f"{', '.join(required_skills[:5])} required")
        
        # Location and work mode
        location = job_data.get('location', '')
        remote = job_data.get('remote', False)
        if location:
            work_mode = "Remote" if remote else "Hybrid"
            parts.append(f"{location}, {work_mode}")
        
        return ". ".join(parts)
    
    @staticmethod
    def generate_required_skills_text(job_data: Dict) -> str:
        """
        Generate skill requirements text.
        
        Example:
        "Required: React, Redux, TypeScript. Preferred: NodeJS. Critical: React"
        """
        requirements = job_data.get('requirements', {})
        
        if not isinstance(requirements, dict):
            return ""
        
        parts = []
        
        # Required skills
        required = requirements.get('required_skills', [])
        if required:
            parts.append(f"Required: {', '.join(required)}")
        
        # Preferred skills
        preferred = requirements.get('preferred_skills', [])
        if preferred:
            parts.append(f"Preferred: {', '.join(preferred)}")
        
        # Critical skills
        critical = requirements.get('critical_skills', [])
        if critical:
            parts.append(f"Critical: {', '.join(critical)}")
        
        return ". ".join(parts)
    
    @staticmethod
    def generate_responsibilities_text(job_data: Dict) -> str:
        """
        Generate responsibilities text.
        
        Example:
        "Build UI dashboards, forms, reusable components. 
         Work with APIs, Redux state management. Collaborate with backend teams."
        """
        description = job_data.get('description', '')
        requirements = job_data.get('requirements', {})
        
        parts = []
        
        # Main description (truncated)
        if description:
            desc_clean = description[:500] + "..." if len(description) > 500 else description
            parts.append(desc_clean)
        
        # Responsibilities from requirements
        if isinstance(requirements, dict):
            responsibilities = requirements.get('responsibilities', [])
            if responsibilities:
                parts.append(". ".join(responsibilities[:5]))
        
        return " ".join(parts)
    
    @staticmethod
    def _extract_required_skills(job_data: Dict) -> List[str]:
        """Extract all required skills from job data."""
        requirements = job_data.get('requirements', {})
        
        if not isinstance(requirements, dict):
            return []
        
        skills = []
        skills.extend(requirements.get('required_skills', []))
        skills.extend(requirements.get('critical_skills', []))
        
        return list(set(skills))  # Remove duplicates
