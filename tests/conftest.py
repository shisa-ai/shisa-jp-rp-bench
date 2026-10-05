"""Suite-wide isolation: tests never need credentials, downloads, or network."""

import os
import socket

import pytest


# Set download isolation before dataset imports during collection.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch, tmp_path, request):
    for key in (
        "OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_API_URL",
        "JUDGE_OPENAI_COMPATIBLE_API_KEY", "JUDGE_OPENAI_COMPATIBLE_API_URL",
        "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "COHERE_API_KEY",
        "MISTRAL_API_KEY", "AWS_ACCESS_KEY", "AWS_SECRET_KEY",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "JP_RP_MAX_TOKENS", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
        "SHISA_API_KEY", "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    attempts = []
    request.node._offline_network_attempts = attempts

    def reject_network(*args, **kwargs):
        attempts.append("connection attempted")
        raise RuntimeError("Network access is forbidden in offline tests")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Surface network attempts even if application code or an xfail catches them."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "teardown" and getattr(item, "_offline_network_attempts", []):
        report.outcome = "failed"
        report.longrepr = "Offline test attempted a network connection (possibly caught by application code)."
        if hasattr(report, "wasxfail"):
            del report.wasxfail
