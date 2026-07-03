"""Tests for the control-room UI server (endpoints + SSE event stream)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest


@pytest.fixture
def ui_base_url():
    from bug_classifier.ui.server import create_server

    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _collect_events(base_url: str, run_id: str, timeout: float = 60.0) -> list[dict]:
    events: list[dict] = []
    with urllib.request.urlopen(
        f"{base_url}/api/events/{run_id}", timeout=timeout
    ) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            events.append(event)
            if event["type"] in ("run_complete", "run_error", "stream_end"):
                break
    return events


class TestUIServer:
    def test_index_served(self, ui_base_url: str) -> None:
        with urllib.request.urlopen(ui_base_url + "/", timeout=10) as response:
            body = response.read().decode("utf-8")
        assert "Multi-Agent Control Room" in body
        assert "Agent network" in body

    def test_meta_exposes_examples_and_flags(self, ui_base_url: str) -> None:
        with urllib.request.urlopen(ui_base_url + "/api/meta", timeout=10) as response:
            meta = json.load(response)
        assert meta["dry_run"] is True
        assert meta["confidence_threshold"] == pytest.approx(0.7)
        assert len(meta["examples"]) >= 5
        assert {"label", "report"} <= set(meta["examples"][0])

    def test_classify_requires_report(self, ui_base_url: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post_json(ui_base_url + "/api/classify", {"report": "  "})
        assert excinfo.value.code == 400

    def test_classification_run_streams_full_event_flow(
        self, ui_base_url: str, patch_all_agents_default
    ) -> None:
        started = _post_json(
            ui_base_url + "/api/classify",
            {"report": "Checkout NPE in PaymentService.java:88 for 30% of users."},
        )
        events = _collect_events(ui_base_url, started["run_id"])

        types = [event["type"] for event in events]
        assert types[0] == "run_started"
        assert "run_complete" in types

        started_nodes = {e["node"] for e in events if e["type"] == "node_start"}
        assert {
            "classify_type",
            "assess_severity",
            "identify_component",
            "supervisor",
            "summarize",
            "log_to_notion",
        } <= started_nodes

        final = next(e for e in events if e["type"] == "run_complete")["state"]
        assert final["bug_type"] == "logic"
        assert final["supervisor_verdict"] == "approve"
        assert final["ticket_url"]

    def test_unknown_run_id_is_404(self, ui_base_url: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(ui_base_url + "/api/events/nope", timeout=10)
        assert excinfo.value.code == 404
