"""Backwards-compatible entrypoint for JP RP Bench absolute evaluation.

This module delegates to ``judge_conversations.py``, which implements the
Gemini-based absolute evaluator without any external orchestration framework.
"""

from judge_conversations import main


if __name__ == "__main__":
    main()
