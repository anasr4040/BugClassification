"""Tests for config helpers (Notion database ID normalization)."""

from __future__ import annotations

import pytest

from bug_classifier.config import normalize_notion_database_id

DB_ID = "361092537d94804cb986c7634b316d68"


class TestNormalizeNotionDatabaseId:
    def test_full_app_notion_url_with_view(self) -> None:
        url = f"https://app.notion.com/p/{DB_ID}?v=361092537d9480bab24f000c4f3c2f3a"
        assert normalize_notion_database_id(url) == DB_ID

    def test_workspace_url_with_slug(self) -> None:
        url = f"https://www.notion.so/myteam/Bug-Tickets-{DB_ID}?v=abc"
        assert normalize_notion_database_id(url) == DB_ID

    def test_bare_id_passthrough(self) -> None:
        assert normalize_notion_database_id(DB_ID) == DB_ID

    def test_hyphenated_uuid(self) -> None:
        hyphenated = "36109253-7d94-804c-b986-c7634b316d68"
        assert normalize_notion_database_id(hyphenated) == DB_ID

    def test_url_without_id_raises(self) -> None:
        with pytest.raises(RuntimeError):
            normalize_notion_database_id("https://app.notion.com/p/not-a-database")
