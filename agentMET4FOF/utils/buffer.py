"""This module contains the buffer classes utilized by the agents

It contains the following classes:

- :class:`AgentBuffer`: Buffer class which is instantiated in every agent to store data
  incrementally
- :class:`MetrologicalAgentBuffer`: Buffer class which is instantiated in every
  metrological agent to store data
"""

import copy
from typing import Dict, Iterable, List, Optional, Sized, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

__all__ = ["AgentBuffer", "MetrologicalAgentBuffer"]

from time_series_buffer import TimeSeriesBuffer


class AgentBuffer:
    """Buffer class which is instantiated in every agent to store data incrementally

    This buffer is necessary to handle multiple inputs coming from agents.

    We can access the buffer like a dict with exposed functions such as .values(),
    .keys() and .items(). The actual dict object is stored in the variable
    :attr:`buffer`.

    Attributes
    ----------
    buffer : dict of iterables or dict of dicts of iterables
        The buffer can be a dict of iterables, or a dict of dict of iterables for nested
        named data. The keys are the names of agents.
    buffer_size :
        The total number of elements to be stored in the agent :attr:`buffer`
    supported_datatypes : list of types
        List of all types supported and thus properly handled by the buffer. Defaults to
        :class:`np.ndarray <NumPy:numpy.ndarray>`, list and Pandas
        :class:`DataFrame <Pandas:pandas.DataFrame>`
    """

    def __init__(self, buffer_size: int = 1000):
        """Initialise a new agent buffer object

        Parameters
        ----------
        buffer_size: int
            Length of buffer allowed.
        """
        self.buffer = {}
        self.buffer_size = buffer_size
        self.supported_datatypes = [list, pd.DataFrame, np.ndarray]

    def __getitem__(self, key):
        return self.buffer[key]

    def check_supported_datatype(self, obj: object) -> bool:
        """Checks whether `value` is an object of one of the supported data types

        Parameters
        ----------
        obj : object
            Value to be checked

        Returns
        ------
        result : boolean
            True if value is an object of one of the supported data types, False if not
        """
        for supported_datatype in self.supported_datatypes:
            if isinstance(obj, supported_datatype):
                return True
        return False

    def update(
        self,
        agent_from: Union[Dict[str, Union[np.ndarray, list, pd.DataFrame]], str],
        data: Union[np.ndarray, list, pd.DataFrame, float, int] = None,
    ):
        """Overrides data in the buffer dict keyed by ``agent_from`` with value ``data``

        If ``data`` is a single value, this converts it into a list first before storing
        in the buffer dict.

        Parameters
        ----------
        agent_from : str
            Name of agent sender
        data : np.ndarray, DataFrame, list, float or int
            New incoming data
        """
        # handle if data type nested in dict
        if isinstance(data, dict):
            # check for each value datatype
            for key, value in data.items():
                # if the value is not list types, turn it into a list of single value
                # i.e [value]
                if not self.check_supported_datatype(value):
                    data[key] = [value]
        elif not self.check_supported_datatype(data):
            data = [data]
        self.buffer.update({agent_from: data})
        return self.buffer

    def _concatenate(
        self,
        iterable: Union[np.ndarray, list, pd.DataFrame],
        data: Union[np.ndarray, list, DataFrame],
        concat_axis: int = 0,
    ) -> Iterable:
        """Concatenate the given ``iterable`` with ``data``

        Handles the concatenation function depending on the datatype, and truncates it
        if the buffer is filled to `buffer_size`.

        Parameters
        ----------
        iterable : any in supported_datatype
            The current buffer to be concatenated with.

        data : np.ndarray, DataFrame, list
            New incoming data

        Returns
        -------
        any in supported_datatype
            the original buffer with the data appended
        """
        # handle list
        if isinstance(iterable, list):
            iterable += data
            # check if exceed memory buffer size, remove the first n elements which
            # exceeded the size
            if len(iterable) > self.buffer_size:
                truncated_element_index = len(iterable) - self.buffer_size
                iterable = iterable[truncated_element_index:]

        # handle if data type is np.ndarray
        elif isinstance(iterable, np.ndarray):
            iterable = np.concatenate((iterable, data), axis=concat_axis)
            if len(iterable) > self.buffer_size:
                truncated_element_index = len(iterable) - self.buffer_size
                iterable = iterable[truncated_element_index:]

        # handle if data type is pd.DataFrame
        elif isinstance(iterable, pd.DataFrame):
            iterable = pd.concat([iterable, data], ignore_index=True, axis=concat_axis)
            if len(iterable) > self.buffer_size:
                truncated_element_index = len(iterable) - self.buffer_size
                iterable = iterable.truncate(before=truncated_element_index)
        return iterable

    def buffer_filled(self, agent_from: Optional[str] = None) -> bool:
        """Checks whether buffer is filled, by comparing against the :attr:`buffer_size`

        For nested dict, this returns True if any of the iterables is beyond the
        :attr:`buffer_size`. For any of the dict values , which is not one of
        :attr:`supported_datatypes` this returns None.

        Parameters
        ----------
        agent_from : str, optional
            Name of input agent in the buffer dict to be looked up. If ``agent_from``
            is not provided, we check for all iterables in the buffer (default).

        Returns
        -------
        bool or None
            True if either the or any of the iterables has reached
            :attr:`buffer_size` or None in case none of the values is of one of the
            supported datatypes. False if all present iterable can take at least
            one more element according to :attr:`buffer_size`.
        """
        if agent_from is None:
            return any([self._iterable_filled(iterable) for iterable in self.values()])
        elif isinstance(self[agent_from], dict):
            return any(
                [
                    self._iterable_filled(iterable)
                    for iterable in self[agent_from].values()
                ]
            )
        else:
            return self._iterable_filled(self[agent_from])

    def _iterable_filled(self, iterable: Sized) -> Union[bool, None]:
        """Internal method for checking on length of iterables of supported types

        Parameters
        ----------
        iterable : Any
            Expected to be an iterable of one of the supported datatypes but could be
            any.

        Returns
        -------
        bool or None
            True if the iterable is of one of the supported datatypes and has reached
            :attr:`buffer_size` in length or False if not or None in case it is not of
            one of the supported datatypes.
        """
        if self.check_supported_datatype(iterable):
            if len(iterable) >= self.buffer_size:
                return True
            return False

    def popleft(self, n: Optional[int] = 1) -> Union[Dict, np.ndarray, list, DataFrame]:
        """Pops the first n entries in the buffer

        Parameters
        ---------
        n : int
            Number of elements to retrieve from buffer

        Returns
        -------
        dict, :class:`np.ndarray <NumPy:numpy.ndarray>`, list or Pandas
        :class:`DataFrame <Pandas:pandas.DataFrame>`
            The retrieved elements
        """
        popped_buffer = copy.copy(self.buffer)
        remaining_buffer = copy.copy(self.buffer)
        if isinstance(popped_buffer, dict):
            for key, value in popped_buffer.items():
                value, remaining_buffer[key] = self._popleft(value, n)
        else:
            popped_buffer, remaining_buffer = self._popleft(popped_buffer, n)
        self.buffer = remaining_buffer
        return popped_buffer

    @staticmethod
    def _popleft(
        iterable: Union[np.ndarray, list, DataFrame], n: Optional[int] = 1
    ) -> Tuple[Union[np.ndarray, list, DataFrame], Union[np.ndarray, list, DataFrame]]:
        """Internal handler of the actual popping mechanism based on type of iterable

        Parameters
        ---------
        n : int
            Number of elements to retrieve from buffer.
        iterable : any in :attr:`supported_datatypes`
            The current buffer to retrieve from.

        Returns
        -------
        2-tuple of each either one of :class:`np.ndarray <NumPy:numpy.ndarray>`,
        list or Pandas :class:`DataFrame <Pandas:pandas.DataFrame>`
            The retrieved elements and the residual items in the buffer
        """
        popped_item = 0
        if isinstance(iterable, list):
            popped_item = iterable[:n]
            iterable = iterable[n:]
        elif isinstance(iterable, np.ndarray):
            popped_item = iterable[:n]
            iterable = iterable[n:]
        elif isinstance(iterable, pd.DataFrame):
            popped_item = iterable.iloc[:n]
            iterable = iterable.iloc[n:]
        return popped_item, iterable

    def clear(self, agent_from: Optional[str] = None):
        """Clears the data in the buffer

        Parameters
        ----------
        agent_from : str, optional
            Name of agent, if ``agent_from`` is not given, the entire buffer is
            flushed. (default)
        """
        if agent_from is None:
            self.buffer = {}
        elif agent_from in self.buffer:
            del self.buffer[agent_from]

    def store(
        self,
        agent_from: Union[Dict[str, Union[np.ndarray, list, pd.DataFrame]], str],
        data: Union[np.ndarray, list, pd.DataFrame, float, int] = None,
        concat_axis: Optional[int] = 0,
    ):
        """Stores data into :attr:`buffer` with the received message

        Checks if sender agent has sent any message before. If it did, then append,
        otherwise create new entry for it.

        Parameters
        ----------
        agent_from : dict | str
            if type is dict, we expect it to be the agentMET4FOF dict message to be
            compliant with older code (keys ``from`` and ``data`` present'), otherwise
            we expect it to be name of agent sender and ``data`` will need to be passed
            as parameter
        data : np.ndarray, DataFrame, list, float or int
            Not used if ``agent_from`` is a dict. Otherwise ``data`` is compulsory.
        concat_axis : int, optional
            axis to concatenate on with the buffering for numpy arrays.
            Default is 0.
        """
        # Store into a separate variables, it will be used frequently later for the
        # type checks. If first argument is the agentMET4FOF dict message in old format
        if isinstance(agent_from, dict):
            message_from = agent_from["from"]
            message_data = agent_from["data"]
        # ... otherwise, we expect the name of agent_sender and the data to be passed.
        else:
            message_from = agent_from
            message_data = data

        # check if sender agent has sent any message before:
        # if it did,then append, otherwise create new entry for the input agent
        if message_from not in self.buffer:
            self.update(message_from, message_data)
            return 0

        # otherwise 'sender' exists in memory, handle appending
        # acceptable data types : list, dict, ndarray, dataframe, single values

        # handle nested data in dict
        if isinstance(message_data, dict):
            for key, value in message_data.items():
                # if it is a single value, then we convert it into a single element list
                if not self.check_supported_datatype(value):
                    value = [value]
                # check if the key exist
                # if it does, then append
                if key in self.buffer[agent_from].keys():
                    self.buffer[agent_from][key] = self._concatenate(
                        self.buffer[agent_from][key], value, concat_axis
                    )
                # otherwise, create new entry
                else:
                    self.buffer[agent_from].update({key: value})
        else:
            if not self.check_supported_datatype(message_data):
                message_data = [message_data]
            self.buffer[agent_from] = self._concatenate(
                self.buffer[agent_from], message_data, concat_axis
            )

    def values(self):
        """Interface to access the internal dict's values()"""
        return self.buffer.values()

    def items(self):
        """Interface to access the internal dict's items()"""
        return self.buffer.items()

    def keys(self):
        """Interface to access the internal dict's keys()"""
        return self.buffer.keys()


class MetrologicalAgentBuffer(AgentBuffer):
    """Buffer class which is instantiated in every metrological agent to store data

    This buffer is necessary to handle multiple inputs coming from agents.

    We can access the buffer like a dict with exposed functions such as .values(),
    .keys() and .items(). The actual dict object is stored in the attribute
    :attr:`buffer <agentMET4FOF.agents.AgentBuffer.buffer>`. The list in
    :attr:`supported_datatypes <agentMET4FOF.agents.AgentBuffer.supported_datatypes>`
    contains one more element
    for metrological agents, namely :class:`TimeSeriesBuffer
    <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>`.
    """

    def __init__(self, buffer_size: int = 1000):
        """Initialise a new agent buffer object

        Parameters
        ----------
        buffer_size: int
            Length of buffer allowed.
        """
        super(MetrologicalAgentBuffer, self).__init__(buffer_size)
        self.supported_datatypes.append(TimeSeriesBuffer)

    def convert_single_to_tsbuffer(self, single_data: Union[List, Tuple, np.ndarray]):
        """Convert common data in agentMET4FOF to :class:`TimeSeriesBuffer
        <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>`

        Parameters
        ----------
        single_data : iterable of iterables (list, tuple, np.ndarray) with shape (N, M)

            * M==2 (pairs): assumed to be like (time, value)
            * M==3 (triple): assumed to be like (time, value, value_unc)
            * M==4 (4-tuple): assumed to be like (time, time_unc, value, value_unc)

        Returns
        -------
        TimeSeriesBuffer
            the new :class:`TimeSeriesBuffer
            <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>` object

        """
        ts = TimeSeriesBuffer(maxlen=self.buffer_size)
        ts.add(single_data)
        return ts

    def update(
        self,
        agent_from: str,
        data: Union[Dict, List, Tuple, np.ndarray],
    ) -> TimeSeriesBuffer:
        """Overrides data in the buffer dict keyed by `agent_from` with value `data`

        Parameters
        ----------
        agent_from : str
            Name of agent sender
        data : dict or iterable of iterables (list, tuple, np.ndarray) with shape (N, M
            the data to be stored in the metrological buffer

        Returns
        -------
        TimeSeriesBuffer
            the updated :class:`TimeSeriesBuffer
            <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>` object
        """
        # handle if data type nested in dict
        if isinstance(data, dict):
            # check for each value datatype
            for key, value in data.items():
                data[key] = self.convert_single_to_tsbuffer(value)
        else:
            data = self.convert_single_to_tsbuffer(data)
            self.buffer.update({agent_from: data})
        return self.buffer

    def _concatenate(
        self,
        iterable: TimeSeriesBuffer,
        data: Union[np.ndarray, list, pd.DataFrame],
        concat_axis: int = 0,
    ) -> TimeSeriesBuffer:
        """Concatenate the given ``TimeSeriesBuffer`` with ``data``

        Add ``data`` to the :class:`TimeSeriesBuffer
        <time-series-buffer:time_series_buffer.buffer.TimeSeriesBuffer>` object.

        Parameters
        ----------
        iterable : TimeSeriesBuffer
            The current buffer to be concatenated with.
        data : np.ndarray, DataFrame, list
            New incoming data

        Returns
        -------
        TimeSeriesBuffer
            the original buffer with the data appended
        """
        iterable.add(data)
        return iterable
