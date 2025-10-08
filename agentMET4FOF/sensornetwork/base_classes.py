from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from agentMET4FOF.agents.metrological_base_agents import MetrologicalAgent
from agentMET4FOF.streams.metrological_base_streams import MetrologicalDataStreamMET4FOF


class SensorOnPlatform(MetrologicalDataStreamMET4FOF):
    """Streaming data from a sensor located on a specified platform

    Parameters
    ----------
    platform_name : str, optional
        name of the platform on which the sensing unit is located
    uncertainty : float
        frequency of wave function, defaults to 50.0
    output_unit : str
        SI unit of the sensor output
    sensor_type : str, optional
        type of sensor based on what is being measured
    data_stream : Union[List, DataFrame, np.ndarray]
        data stream of sensor measurements indexed by time, e.g. timestamps
    """

    def __init__(
        self, uncertainty: float =0, platform_name=None, sensor_type=None, output_unit=None, data_stream: Union[List, pd.DataFrame, np.ndarray]=None
    ):
        self.uncertainty = uncertainty
        self.output_unit = output_unit
        self.platform = platform_name
        self.sensor_type = sensor_type
        super(SensorOnPlatform, self).__init__(value_unc=self.uncertainty, time_unc=0)
        self.set_metadata(
            self.platform+'_'+data_stream.columns.values[0],
            "time",
            "h",
            self.sensor_type,
            output_unit,
            "Data Stream from Heat Meter Readings",
        )
        self.set_data_source(quantities=data_stream, time=pd.DataFrame(data_stream.index.values))


class SensorPlatform(MetrologicalAgent):
    """A metrological agent representing a platform hosting one or more sensors in an IoT network
     """

    def init_parameters(self):
        """Initialize the sensor agent

         Parameters
         -----------
          uncertainty: np.float
            The uncertainty of the sensor determined via a calibration
          position: Union[Tuple[np.float, np.float], str]
            The location of the sensor  given either by explicit geographical coordinates or a string descriptor
         """

        super().init_parameters()
        self.position = None
        self.output_unit = None
        self._stream = SensorOnPlatform(uncertainty=.01, platform_name="Heat Meter", sensor_type='Temperature', output_unit='°C',
                                        data_stream=None)

    @property
    def device_id(self):
        return self._stream.metadata.metadata["device_id"]

