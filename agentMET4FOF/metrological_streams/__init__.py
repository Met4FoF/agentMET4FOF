"""This module ensures backwards compatibility of the module to package refactoring

Up until version 0.8.1 we had several modules instead of the packages `agents` and
`streams`. We have included this package to not break the previously needed import
statements.
"""
from ..streams.metrological_base_streams import MetrologicalDataStreamMET4FOF
from ..streams.signal_streams import (
    MetrologicalMultiWaveGenerator,
    MetrologicalSineGenerator,
)

__all__ = [
    "MetrologicalMultiWaveGenerator",
    "MetrologicalSineGenerator",
    "MetrologicalDataStreamMET4FOF",
]
