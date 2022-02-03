"""This package contains utility modules and classes for the core classes"""

__all__ = ["AgentBuffer", "MetrologicalAgentBuffer", "Backend"]

from enum import auto, Enum

from .buffer import AgentBuffer, MetrologicalAgentBuffer


class Backend(Enum):
    OSBRAIN = auto()
    MESA = auto()
