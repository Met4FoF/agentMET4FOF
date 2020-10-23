import numpy as np
import pandas as pd
import pytest

from agentMET4FOF.agents import AgentMET4FOF, MonitorAgent
from tests.conftest import test_timeout

num_samples = 10

# prepare dummy data stream
datastream_x = list(np.arange(num_samples))
datastream_y = list(np.arange(num_samples))
datastream_y.reverse()

print(datastream_x)
print(datastream_y)


# test different types of message type & see how the agent handles the memory storage
class SingleValueAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {"x": datastream_x, "y": datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = self.stream["x"][self.pointer]
                self.pointer += 1
                self.send_output(data)


class ListAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {"x": datastream_x, "y": datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = [self.stream["x"][self.pointer]]
                self.pointer += 1
                self.send_output(data)


class NpArrayAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {"x": datastream_x, "y": datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = np.array([self.stream["x"][self.pointer]])
                self.pointer += 1
                self.send_output(data)


class PdDataFrameAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {"x": datastream_x, "y": datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = pd.DataFrame.from_dict(self.stream).iloc[
                    self.pointer : self.pointer + 1
                ]
                self.pointer += 1
                self.send_output(data)


class NestedDict_SingleValueAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {"x": datastream_x, "y": datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data_x = self.stream["x"][self.pointer : self.pointer + 1]
                data_y = self.stream["y"][self.pointer : self.pointer + 1]
                self.pointer += 1
                self.send_output({"x": data_x, "y": data_y})


class NestedDict_ListAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {"x": datastream_x, "y": datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data_x = self.stream["x"][self.pointer : self.pointer + 1]
                data_y = self.stream["y"][self.pointer : self.pointer + 1]
                self.pointer += 1
                self.send_output({"x": data_x, "y": data_y})


class NestedDict_NpArrayAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {"x": datastream_x, "y": datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data_x = np.array(self.stream["x"][self.pointer : self.pointer + 1])
                data_y = np.array(self.stream["y"][self.pointer : self.pointer + 1])
                self.pointer += 1
                self.send_output({"x": data_x, "y": data_y})


params = [
    (SingleValueAgent, {"SingleValueAgent_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),
    (ListAgent, {"ListAgent_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),
    (NpArrayAgent, {"NpArrayAgent_1": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}),
    (
        PdDataFrameAgent,
        {
            "PdDataFrameAgent_1": pd.DataFrame.from_dict(
                {"x": datastream_x, "y": datastream_y}
            )
        },
    ),
    (
        NestedDict_SingleValueAgent,
        {
            "NestedDict_SingleValueAgent_1": {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "y": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            }
        },
    ),
    (
        NestedDict_ListAgent,
        {
            "NestedDict_ListAgent_1": {
                "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "y": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            }
        },
    ),
    (
        NestedDict_NpArrayAgent,
        {
            "NestedDict_NpArrayAgent_1": {
                "x": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                "y": np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
            }
        },
    ),
]


@pytest.mark.timeout(test_timeout)
@pytest.mark.parametrize("agentType,expected_monitor_results", params)
def test_simpleAgent(agent_network, agentType, expected_monitor_results):
    # init agents by adding into the agent network
    simple_agent = agent_network.add_agent(agentType=agentType)
    monitor_agent_1 = agent_network.add_agent(agentType=MonitorAgent)

    # shorten n wait loop time
    simple_agent.init_agent_loop(0.01)

    # connect agents
    agent_network.bind_agents(simple_agent, monitor_agent_1)

    # set all agents states to "Running"
    agent_network.set_running_state()

    # Run check of expected and actual result until test times out.
    is_not_expected = True
    while is_not_expected:
        try:
            # Run actual check. This reduces test runtime in case of passed tests but
            # results in quite cryptic error messages in case it fails due to the
            # timeout causing the actual fail. So, if this line fails, regardless of
            # the error message, it means, the addressed attribute's content does not
            # match the expected expression.
            assert str(monitor_agent_1.get_attr("buffer").buffer) == str(
                expected_monitor_results
            )
            # End test execution, if test passes.
            is_not_expected = False
        except AssertionError:
            pass
