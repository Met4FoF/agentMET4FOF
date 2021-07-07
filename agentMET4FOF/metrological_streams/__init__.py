"""This package ensures backwards compatibility of the module to package refactoring

Up until version 0.8.1 we had several modules instead of the packages `agents` and
`streams`. We have included this package to not break the previously needed import
statements.
"""
import warnings

from ..streams.metrological_base_streams import MetrologicalDataStreamMET4FOF
from ..streams.metrological_signal_streams import (
    MetrologicalMultiWaveGenerator,
    MetrologicalSineGenerator,
)

__all__ = [
    "MetrologicalDataStreamMET4FOF",
    "MetrologicalMultiWaveGenerator",
    "MetrologicalSineGenerator",
]

warnings.warn(
    "The package metrological_streams is deprecated and might be removed any "
    "time. The content is moved into the package streams, such that instead "
    "for instance 'from agentMET4FOF.metrological_streams import "
    "MetrologicalDataStreamMET4FOF' the current import should be 'from "
    "agentMET4FOF.streams.metrological_base_streams import "
    "MetrologicalDataStreamMET4FOF' or 'from "
    "agentMET4FOF.streams import "
    "MetrologicalDataStreamMET4FOF'.",
    DeprecationWarning,
)
