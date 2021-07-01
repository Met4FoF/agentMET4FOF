"""This module ensures backwards compatibility of the module to package refactoring

Up until version 0.8.1 we had several modules instead of the packages `agents` and
`streams`. We have included this package to not break the previously needed import
statements.
"""
from .base_agents import AgentMET4FOF, MonitorAgent
from .metrological_base_agents import (
    MetrologicalAgent,
    MetrologicalMonitorAgent,
)
from .metrological_signal_agents import MetrologicalGeneratorAgent
from .network import AgentNetwork
from .signal_agents import SineGeneratorAgent
from .utils import AgentBuffer, MetrologicalAgentBuffer
