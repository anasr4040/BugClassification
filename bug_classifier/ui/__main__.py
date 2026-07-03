"""Launch the bug classifier control-room UI.

Usage:
    python -m bug_classifier.ui                 # http://127.0.0.1:8765
    python -m bug_classifier.ui --port 9000
    python -m bug_classifier.ui --dry-run       # force NOTION_DRY_RUN=true
    python -m bug_classifier.ui --no-browser
"""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Web UI for the multi-agent bug classifier."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Port (default 8765).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force NOTION_DRY_RUN=true (print ticket payloads, no Notion writes).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        os.environ["NOTION_DRY_RUN"] = "true"

    # Import after env overrides so config picks up --dry-run.
    from bug_classifier.ui.server import serve

    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
