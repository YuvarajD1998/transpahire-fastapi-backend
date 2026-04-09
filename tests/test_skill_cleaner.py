import pytest
from app.services.file_service import SkillCleaner


@pytest.fixture
def cleaner():
    return SkillCleaner()


def make_skill(name, context=None):
    return {"name": name, "context": context}


class TestVerbPrefixes:
    def test_verb_prefix_rejected(self, cleaner):
        skills = [make_skill("managing projects")]
        assert cleaner.clean(skills) == []

    def test_multiple_verb_prefixes_rejected(self, cleaner):
        skills = [
            make_skill("managing teams"),
            make_skill("handling escalations"),
            make_skill("leading sprints"),
        ]
        assert cleaner.clean(skills) == []

    def test_non_verb_prefix_kept(self, cleaner):
        skills = [make_skill("React")]
        assert len(cleaner.clean(skills)) == 1


class TestBlocklist:
    def test_blocklist_generic_testing_removed(self, cleaner):
        skills = [make_skill("Testing")]
        assert cleaner.clean(skills) == []

    def test_qualified_testing_kept(self, cleaner):
        skills = [make_skill("Regression Testing")]
        result = cleaner.clean(skills)
        assert len(result) == 1
        assert result[0]["name"] == "Regression Testing"

    def test_blocklist_management_removed(self, cleaner):
        skills = [make_skill("Management")]
        assert cleaner.clean(skills) == []

    def test_qualified_management_kept(self, cleaner):
        skills = [make_skill("Vendor Management")]
        result = cleaner.clean(skills)
        assert len(result) == 1

    def test_soft_trait_removed(self, cleaner):
        skills = [make_skill("team player"), make_skill("hardworking")]
        assert cleaner.clean(skills) == []


class TestWordCountLimit:
    def test_over_five_words_rejected(self, cleaner):
        skills = [make_skill("experience with distributed cloud systems here")]
        assert cleaner.clean(skills) == []

    def test_five_words_kept(self, cleaner):
        skills = [make_skill("AWS Lambda Function URL")]
        result = cleaner.clean(skills)
        assert len(result) == 1


class TestDeduplication:
    def test_react_and_reactjs_deduplicated(self, cleaner):
        skills = [make_skill("React"), make_skill("React.js")]
        result = cleaner.clean(skills)
        assert len(result) == 1

    def test_prefers_entry_with_context(self, cleaner):
        skills = [
            make_skill("React", context=None),
            make_skill("React.js", context="built UI with React.js"),
        ]
        result = cleaner.clean(skills)
        assert len(result) == 1
        assert result[0]["context"] == "built UI with React.js"

    def test_first_entry_kept_when_no_context(self, cleaner):
        skills = [make_skill("Python"), make_skill("Python")]
        result = cleaner.clean(skills)
        assert len(result) == 1
        assert result[0]["name"] == "Python"


class TestCap:
    def test_cap_at_40(self, cleaner):
        skills = [make_skill(f"Skill{i}") for i in range(50)]
        result = cleaner.clean(skills)
        assert len(result) == 40

    def test_under_40_not_truncated(self, cleaner):
        skills = [make_skill(f"Skill{i}") for i in range(20)]
        result = cleaner.clean(skills)
        assert len(result) == 20


class TestEmptyInput:
    def test_empty_list(self, cleaner):
        assert cleaner.clean([]) == []

    def test_skill_with_no_name(self, cleaner):
        skills = [{"name": "", "context": None}, {"name": None, "context": None}]
        assert cleaner.clean(skills) == []
