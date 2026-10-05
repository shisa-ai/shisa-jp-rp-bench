"""API-based Japanese roleplay generation and absolute scoring.

Dataset loading and benchmark orchestration live in their explicit modules so
importing API clients or artifact utilities does not load dataset dependencies.
"""
from .models import create_client, generate_response, generate_completion

__all__ = ["create_client", "generate_response", "generate_completion"]
