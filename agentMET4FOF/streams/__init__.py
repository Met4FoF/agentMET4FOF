"""This module ensures backwards compatibility of the module to package refactoring

Up until version 0.8.1 we had several modules instead of the packages `agents` and
`streams`. We have included this package to not break the previously needed import
statements.
"""
import warnings

from .base_streams import DataStreamMET4FOF
from .metrological_base_streams import MetrologicalDataStreamMET4FOF
from .metrological_signal_streams import (
    MetrologicalMultiWaveGenerator,
    MetrologicalSineGenerator,
)
from .signal_streams import CosineGenerator, SineGenerator

__all__ = [
    "CosineGenerator",
    "DataStreamMET4FOF",
    "MetrologicalDataStreamMET4FOF",
    "MetrologicalMultiWaveGenerator",
    "MetrologicalSineGenerator",
    "SineGenerator",
]


warnings.warn(
    "The package metrological_agents is deprecated and might be removed any "
    "time. The content is moved into the package agents, such that instead "
    "for instance 'from agentMET4FOF.metrological_agents import "
    "MetrologicalAgent' the current import should be 'from "
    "agentMET4FOF.agents.metrological_agents import MetrologicalAgent'.",
    DeprecationWarning,
)
