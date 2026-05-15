"""Notion integration for persisting classified bug tickets.

Direct SDK approach
-------------------
This module calls the Notion REST API via ``notion-client``. The graph's
``log_to_notion`` node maps ``BugState`` fields onto database properties and
creates a page.

MCP alternative (decoupled integrations)
----------------------------------------
An MCP server could expose tool schemas such as ``create_bug_ticket`` and
``validate_database_schema``. LangGraph agents would invoke those tools through
LangChain's tool interface (``@tool`` + ``bind_tools``) instead of importing
``notion-client`` directly. The MCP server would translate each tool call into
Notion API requests, keeping agent/orchestration logic independent of Notion's
SDK and auth wiring. Swapping Notion for Jira or Linear would mean changing the
MCP server only, not the classification agents.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from notion_client import Client
from notion_client.errors import (
    APIErrorCode,
    APIResponseError,
    RequestTimeoutError,
    is_notion_client_error,
)

from bug_classifier import config
from bug_classifier.state import BugState

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0

PROPERTY_TITLE = "Title"
PROPERTY_BUG_TYPE = "Bug Type"
PROPERTY_SEVERITY = "Severity"
PROPERTY_COMPONENT = "Component"
PROPERTY_SUMMARY = "Summary"
PROPERTY_STATUS = "Status"
PROPERTY_CONFIDENCE = "Confidence"
PROPERTY_CREATED_BY = "Created By"

BUG_TYPE_OPTIONS = ("Crash", "Performance", "Logic", "Security", "Ui")
SEVERITY_OPTIONS = ("P0", "P1", "P2", "P3")
STATUS_OPTIONS = ("New",)

_data_source_id_cache: dict[str, str] = {}

_TRANSIENT_API_CODES = frozenset(
    {
        APIErrorCode.RateLimited,
        APIErrorCode.InternalServerError,
        APIErrorCode.ServiceUnavailable,
        APIErrorCode.GatewayTimeout,
        APIErrorCode.ConflictError,
    }
)


def _get_client() -> Client:
    return Client(auth=config.NOTION_API_KEY)


def _component_options() -> tuple[str, ...]:
    return tuple(name.capitalize() for name in config.BUG_COMPONENTS)


def _database_schema_properties() -> dict[str, Any]:
    """Return the full property schema for first-time database setup."""
    return {
        PROPERTY_TITLE: {"title": {}},
        PROPERTY_BUG_TYPE: {
            "select": {
                "options": [{"name": name, "color": "default"} for name in BUG_TYPE_OPTIONS]
            }
        },
        PROPERTY_SEVERITY: {
            "select": {
                "options": [
                    {"name": "P0", "color": "red"},
                    {"name": "P1", "color": "orange"},
                    {"name": "P2", "color": "yellow"},
                    {"name": "P3", "color": "green"},
                ]
            }
        },
        PROPERTY_COMPONENT: {
            "select": {
                "options": [
                    {"name": name, "color": "default"} for name in _component_options()
                ]
            }
        },
        PROPERTY_SUMMARY: {"rich_text": {}},
        PROPERTY_STATUS: {
            "select": {
                "options": [{"name": "New", "color": "blue"}]
            }
        },
        PROPERTY_CONFIDENCE: {"number": {"format": "number"}},
        PROPERTY_CREATED_BY: {"rich_text": {}},
    }


def _capitalize_option(value: str) -> str:
    return value.strip().capitalize()


def _average_confidence(scores: dict[str, float] | None) -> float | None:
    if not scores:
        return None
    values = list(scores.values())
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _rich_text(content: str) -> dict[str, Any]:
    return {"rich_text": [{"type": "text", "text": {"content": content[:2000]}}]}


def _title_text(content: str, *, urgent: bool = False) -> str:
    prefix = "[P0 URGENT] " if urgent else ""
    return f"{prefix}{content}"[:100]


def _build_page_properties(
    state: BugState, *, urgent: bool = False, title_property: str = PROPERTY_TITLE
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        title_property: {
            "title": [
                {
                    "type": "text",
                    "text": {"content": _title_text(state["raw_report"], urgent=urgent)},
                }
            ]
        },
        PROPERTY_STATUS: {"select": {"name": "New"}},
        PROPERTY_CREATED_BY: _rich_text("AI Bug Classifier"),
    }

    if state.get("bug_type"):
        properties[PROPERTY_BUG_TYPE] = {
            "select": {"name": _capitalize_option(state["bug_type"])}
        }

    if state.get("severity"):
        properties[PROPERTY_SEVERITY] = {"select": {"name": state["severity"]}}

    if state.get("component"):
        properties[PROPERTY_COMPONENT] = {
            "select": {"name": _capitalize_option(state["component"])}
        }

    summary = state.get("summary")
    if summary:
        properties[PROPERTY_SUMMARY] = _rich_text(summary)
    elif urgent:
        properties[PROPERTY_SUMMARY] = _rich_text(
            "P0 emergency escalation — standard summary pipeline was bypassed."
        )

    confidence = _average_confidence(state.get("confidence_scores"))
    if confidence is not None:
        properties[PROPERTY_CONFIDENCE] = {"number": confidence}

    return properties


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, RequestTimeoutError):
        return True
    if isinstance(exc, APIResponseError):
        if exc.code in _TRANSIENT_API_CODES:
            return True
        return exc.status >= 500
    if is_notion_client_error(exc):
        return False
    # Network-level errors from httpx bubble up as generic exceptions.
    return exc.__class__.__name__ in {"ConnectError", "ReadTimeout", "WriteTimeout", "NetworkError"}


def _with_retries(operation: Callable[[], Any], *, description: str) -> Any:
    delay = INITIAL_BACKOFF_SECONDS
    last_error: BaseException | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES or not _is_transient_error(exc):
                raise
            logger.warning(
                "Transient Notion error during %s (attempt %d/%d): %s. Retrying in %.1fs.",
                description,
                attempt,
                MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)
            delay *= 2

    if last_error:
        raise last_error
    raise RuntimeError(f"Notion operation failed: {description}")


def _resolve_data_source_id(client: Client, database_id: str) -> str:
    if database_id in _data_source_id_cache:
        return _data_source_id_cache[database_id]

    database = _with_retries(
        lambda: client.databases.retrieve(database_id),
        description="retrieve database",
    )
    data_sources = database.get("data_sources") or []
    if data_sources:
        data_source_id = data_sources[0]["id"]
    else:
        # Fallback for older API responses.
        data_source_id = database_id

    _data_source_id_cache[database_id] = data_source_id
    return data_source_id


def _fetch_data_source_properties(
    client: Client, database_id: str
) -> tuple[str, dict[str, Any]]:
    data_source_id = _resolve_data_source_id(client, database_id)
    data_source = _with_retries(
        lambda: client.data_sources.retrieve(data_source_id),
        description="retrieve data source schema",
    )
    return data_source_id, data_source.get("properties", {})


def _select_option_names(properties: dict[str, Any], property_name: str) -> set[str]:
    schema = properties.get(property_name, {})
    if schema.get("type") != "select":
        return set()
    return {
        option["name"]
        for option in schema.get("select", {}).get("options", [])
        if option.get("name")
    }


def _validate_select_values(
    properties: dict[str, Any], page_properties: dict[str, Any]
) -> list[str]:
    """Ensure select values exist in the database schema before page creation."""
    checks = {
        PROPERTY_BUG_TYPE: page_properties.get(PROPERTY_BUG_TYPE, {})
        .get("select", {})
        .get("name"),
        PROPERTY_SEVERITY: page_properties.get(PROPERTY_SEVERITY, {})
        .get("select", {})
        .get("name"),
        PROPERTY_COMPONENT: page_properties.get(PROPERTY_COMPONENT, {})
        .get("select", {})
        .get("name"),
        PROPERTY_STATUS: page_properties.get(PROPERTY_STATUS, {})
        .get("select", {})
        .get("name"),
    }

    errors: list[str] = []
    for property_name, value in checks.items():
        if not value:
            continue
        available = _select_option_names(properties, property_name)
        if available and value not in available:
            errors.append(
                f"{property_name!r} value {value!r} is not configured in Notion. "
                f"Available options: {sorted(available)}"
            )
    return errors


def _print_dry_run(
    state: BugState, page_properties: dict[str, Any], *, urgent: bool = False
) -> None:
    payload = {
        "database_id": config.NOTION_DATABASE_ID,
        "urgent": urgent,
        "properties": page_properties,
        "raw_state": {
            "bug_type": state.get("bug_type"),
            "severity": state.get("severity"),
            "component": state.get("component"),
            "confidence_scores": state.get("confidence_scores"),
        },
    }
    print("NOTION_DRY_RUN=true — ticket that WOULD be created:")
    print(json.dumps(payload, indent=2, default=str))


def setup_notion_database(parent_page_id: str | None = None) -> str:
    """Create or validate the bug ticket Notion database.

    If ``NOTION_DATABASE_ID`` already exists and is accessible, ensures required
    properties and select options are present. Otherwise creates a new database
    under ``parent_page_id`` (or ``NOTION_PARENT_PAGE_ID`` from config).

    Returns:
        The database ID (existing or newly created).
    """
    client = _get_client()
    parent = parent_page_id or config.NOTION_PARENT_PAGE_ID

    try:
        _with_retries(
            lambda: client.databases.retrieve(config.NOTION_DATABASE_ID),
            description="check existing database",
        )
        logger.info("Notion database %s already exists; validating schema.", config.NOTION_DATABASE_ID)
        _ensure_database_schema(client, config.NOTION_DATABASE_ID)
        return config.NOTION_DATABASE_ID
    except APIResponseError as exc:
        if exc.code != APIErrorCode.ObjectNotFound:
            raise
        logger.info("Database not found; creating a new bug ticket database.")

    if not parent:
        raise RuntimeError(
            "Cannot create Notion database: set NOTION_PARENT_PAGE_ID in .env or pass "
            "parent_page_id to setup_notion_database()."
        )

    created = _with_retries(
        lambda: client.databases.create(
            parent={"type": "page_id", "page_id": parent},
            title=[{"type": "text", "text": {"content": "Bug Tickets"}}],
            initial_data_source={"properties": _database_schema_properties()},
        ),
        description="create database",
    )
    database_id = created["id"]
    logger.info("Created Notion bug database: %s", database_id)
    return database_id


def _title_property_name(properties: dict[str, Any]) -> str:
    for name, schema in properties.items():
        if schema.get("type") == "title":
            return name
    return PROPERTY_TITLE


def _ensure_database_schema(client: Client, database_id: str) -> dict[str, Any]:
    """Ensure required properties and select options exist; return latest schema."""
    data_source_id, properties = _fetch_data_source_properties(client, database_id)
    desired = _database_schema_properties()
    updates: dict[str, Any] = {}

    for property_name, schema in desired.items():
        if property_name == PROPERTY_TITLE:
            continue
        if property_name not in properties:
            updates[property_name] = schema
            continue
        if schema.get("type") != "select":
            continue
        required_names = {
            option["name"]
            for option in schema.get("select", {}).get("options", [])
            if option.get("name")
        }
        current = _select_option_names(properties, property_name)
        missing = [name for name in required_names if name not in current]
        if missing:
            existing_options = properties[property_name]["select"]["options"]
            merged = existing_options + [
                {"name": name, "color": "default"} for name in missing
            ]
            updates[property_name] = {"select": {"options": merged}}

    if updates:
        _with_retries(
            lambda: client.data_sources.update(data_source_id, properties=updates),
            description="update data source schema",
        )
        logger.info("Updated Notion schema: %s", ", ".join(updates))
        _, properties = _fetch_data_source_properties(client, database_id)

    return properties


def log_to_notion(state: BugState, *, urgent: bool = False) -> dict:
    """Create a Notion database page from classified bug state.

    Args:
        state: Fully or partially classified bug state.
        urgent: When True, prefix the title for P0 emergency escalations.

    Returns:
        ``{"ticket_url": ...}`` on success, or ``{"error": ...}`` on failure.
        Never raises — errors are captured so the pipeline can finish gracefully.
    """
    page_properties = _build_page_properties(state, urgent=urgent)

    if config.NOTION_DRY_RUN:
        _print_dry_run(state, page_properties, urgent=urgent)
        return {"ticket_url": "https://www.notion.so/dry-run-bug-ticket"}

    client = _get_client()
    database_id = config.NOTION_DATABASE_ID

    try:
        schema_properties = _ensure_database_schema(client, database_id)
        title_property = _title_property_name(schema_properties)
        page_properties = _build_page_properties(
            state, urgent=urgent, title_property=title_property
        )
        data_source_id = _resolve_data_source_id(client, database_id)
        validation_errors = _validate_select_values(schema_properties, page_properties)
        if validation_errors:
            message = "; ".join(validation_errors)
            logger.error("Notion select validation failed: %s", message)
            return {"error": message}

        page = _with_retries(
            lambda: client.pages.create(
                parent={"type": "data_source_id", "data_source_id": data_source_id},
                properties=page_properties,
            ),
            description="create Notion page",
        )
        ticket_url = page.get("url", "")
        logger.info("Created Notion bug ticket: %s", ticket_url)
        return {"ticket_url": ticket_url}

    except Exception as exc:
        message = f"Notion API error: {exc}"
        logger.exception("Failed to create Notion bug ticket")
        return {"error": message}


log_classification_to_notion = log_to_notion
