# agents/base_agent.py
from typing import Any, Dict

class agent:
    """Decorator for Agno agents"""
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, cls):
        cls.name = self.name
        return cls