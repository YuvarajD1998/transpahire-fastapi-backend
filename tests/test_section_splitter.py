import pytest
from app.services.file_service import SectionSplitter


@pytest.fixture
def splitter():
    return SectionSplitter()


RESUME_WITH_HEADERS = """\
John Doe
john@example.com

Summary
Experienced software engineer with 5 years in backend development.

Skills
Python, FastAPI, PostgreSQL, Docker

Experience
Software Engineer at Acme Corp 2020-2024
Built REST APIs using FastAPI and PostgreSQL.

Education
B.Tech Computer Science, XYZ University 2016-2020
"""

RESUME_NO_HEADERS = """\
John Doe
john@example.com
Python FastAPI PostgreSQL
Acme Corp 2020-2024
XYZ University 2016-2020
"""


class TestSplitWithHeaders:
    def test_returns_dict_with_known_sections(self, splitter):
        result = splitter.split(RESUME_WITH_HEADERS)
        assert isinstance(result, dict)
        assert "skills" in result or "experience" in result or "education" in result

    def test_section_content_is_correct(self, splitter):
        result = splitter.split(RESUME_WITH_HEADERS)
        if "skills" in result:
            assert "Python" in result["skills"]

    def test_at_least_two_sections_identified(self, splitter):
        result = splitter.split(RESUME_WITH_HEADERS)
        assert "full" not in result, "Should have split into named sections"
        assert len(result) >= 2

    def test_summary_section_captured(self, splitter):
        result = splitter.split(RESUME_WITH_HEADERS)
        assert "summary" in result
        assert "software engineer" in result["summary"].lower()


class TestNoHeaders:
    def test_returns_full_key_when_no_headers(self, splitter):
        result = splitter.split(RESUME_NO_HEADERS)
        assert "full" in result

    def test_full_text_preserved(self, splitter):
        result = splitter.split(RESUME_NO_HEADERS)
        assert result["full"] == RESUME_NO_HEADERS


class TestEdgeCases:
    def test_empty_string_does_not_crash(self, splitter):
        result = splitter.split("")
        assert isinstance(result, dict)

    def test_whitespace_only_does_not_crash(self, splitter):
        result = splitter.split("   \n\n   ")
        assert isinstance(result, dict)

    def test_single_section_header_returns_full(self, splitter):
        text = "Skills\nPython Django"
        result = splitter.split(text)
        # Only one section filled → falls back to full
        assert "full" in result

    def test_case_insensitive_header_matching(self, splitter):
        text = "EXPERIENCE\nSoftware Engineer at Corp\n\nEDUCATION\nB.Tech CS"
        result = splitter.split(text)
        assert "full" not in result


class TestNeedsChunking:
    def test_short_text_does_not_need_chunking(self):
        assert SectionSplitter.needs_chunking("short text", 8000) is False

    def test_long_text_needs_chunking(self):
        long = "x" * 9000
        assert SectionSplitter.needs_chunking(long, 8000) is True

    def test_exact_limit_does_not_need_chunking(self):
        exact = "x" * 8000
        assert SectionSplitter.needs_chunking(exact, 8000) is False
