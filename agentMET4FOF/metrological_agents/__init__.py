"""This package ensures backwards compatibility of the module to package refactoring

Up until version 0.8.1 we had several modules instead of the packages `agents` and
`streams`. We have included this package to not break the previously needed import
statements.
"""
import warnings

from ..agents.metrological_base_agents import (
    MetrologicalAgent,
    MetrologicalMonitorAgent,
)
from ..agents.metrological_signal_agents import MetrologicalGeneratorAgent
from ..utils.buffer import MetrologicalAgentBuffer

__all__ = [
    "MetrologicalAgent",
    "MetrologicalMonitorAgent",
    "MetrologicalAgentBuffer",
    "MetrologicalGeneratorAgent",
]

warnings.warn(
    "The package metrological_agents is deprecated and might be removed any "
    "time. The content is moved into the package agents, such that instead "
    "for instance 'from agentMET4FOF.metrological_agents import "
    "MetrologicalAgent' the current import should be 'from "
    "agentMET4FOF.agents.metrological_agents import MetrologicalAgent'.",
    DeprecationWarning,
)
