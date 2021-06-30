"""This module ensures backwards compatibility of the module to package refactoring

Up until version 0.8.1 we had several modules instead of the packages `agents` and
`streams`. We have included this package to not break the previously needed import
statements.
"""
from ..agents.base_agents import (
    MetrologicalAgent,
    MetrologicalAgentBuffer,
    MetrologicalMonitorAgent,
)

__all__ = ["MetrologicalAgent", "MetrologicalAgentBuffer", "MetrologicalMonitorAgent"]
