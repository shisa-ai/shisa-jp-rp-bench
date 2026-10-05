"""Ensure swallowed network errors cannot make the offline suite look green."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("expected_failure", [False, True])
def test_network_guard_reports_attempts_even_when_caught(tmp_path, expected_failure):
    conftest = Path(__file__).with_name("conftest.py").read_text()
    (tmp_path / "conftest.py").write_text(conftest)
    decorator = '@pytest.mark.xfail(strict=True, reason="known unrelated bug")\n' if expected_failure else ""
    (tmp_path / "test_network.py").write_text(
        "import socket\nimport pytest\n"
        + decorator
        + "def test_swallowed_connection():\n"
        + "    try:\n"
        + "        socket.create_connection(('offline.invalid', 443))\n"
        + "    except Exception:\n"
        + "        pass\n"
        + ("    assert False, 'known unrelated bug'\n" if expected_failure else "")
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Offline test attempted a network connection" in result.stdout
    assert "1 error" in result.stdout
