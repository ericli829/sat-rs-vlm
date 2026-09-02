from __future__ import annotations

import sys
from pathlib import Path

from taskgraph_lab.tools.run_planner_rag_ab import _run


def test_serial_launcher_captures_unicode_without_console_reencoding(tmp_path: Path) -> None:
    log = tmp_path / "child.log"
    return_code = _run(
        [sys.executable, "-c", "print('loading \\u2588 \\ufffd')"],
        log,
    )
    assert return_code == 0
    assert log.read_text(encoding="utf-8").strip() == "loading \u2588 \ufffd"
