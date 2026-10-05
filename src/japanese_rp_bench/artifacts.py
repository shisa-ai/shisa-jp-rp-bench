"""One path and atomic-write contract for benchmark judging artifacts."""
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote


def safe_name(value):
    """Encode names reversibly without allowing path traversal or collisions."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Artifact model/judge names must be nonempty strings")
    encoded = quote(value, safe="-._")
    return encoded.replace(".", "%2E") if encoded in {".", ".."} else encoded


def score_paths(model, judge, output_dir):
    directory = Path(output_dir) / safe_name(model) / safe_name(judge)
    return {"scores": directory / "scores.json", "judgements": directory / "judgements.jsonl", "answers": directory / "answers.jsonl"}


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".pending-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, records):
    atomic_write(path, ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records))
