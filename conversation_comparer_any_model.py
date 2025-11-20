"""Legacy conversation comparison entrypoint.

The original implementation used an external orchestration framework to
analyze conversation pairs. That dependency has been removed from this
repository. If you need the old comparison behavior, please refer to the
git history of this file.
"""

import sys


def main() -> None:
    """Notify callers that the legacy comparer has been removed."""
    sys.stderr.write(
        "conversation_comparer_any_model.py: the legacy comparison\n"
        "pipeline has been removed from this repo. Please use the absolute\n"
        "judge path (judge_conversations.py) or consult git history for the\n"
        "original implementation.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
