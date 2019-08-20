import pytest

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
import numpy as np
import pandas as pd
import time
#init params
np.random.seed(123)
num_samples = 10
test_timeout = 5

#prepare dummy data stream
datastream_x = list(np.arange(num_samples))
datastream_y = list(np.arange(num_samples))
datastream_y.reverse()

print(datastream_x)
print(datastream_y)

#test different types of message type & see how the agent handles the memory storage
class SingleValueAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {'x':datastream_x,'y':datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = self.stream['x'][self.pointer]
                self.pointer += 1
                self.send_output(data)

class ListAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {'x':datastream_x,'y':datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = [self.stream['x'][self.pointer]]
                self.pointer += 1
                self.send_output(data)

class NpArrayAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {'x':datastream_x,'y':datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = np.array([self.stream['x'][self.pointer]])
                self.pointer += 1
                self.send_output(data)

class PdDataFrameAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {'x':datastream_x,'y':datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data = pd.DataFrame.from_dict(self.stream).iloc[self.pointer:self.pointer+1]
                self.pointer += 1
                self.send_output(data)

class NestedDict_SingleValueAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {'x':datastream_x,'y':datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data_x = self.stream['x'][self.pointer:self.pointer+1]
                data_y = self.stream['y'][self.pointer:self.pointer+1]
                self.pointer += 1
                self.send_output({'x':data_x,'y':data_y})

class NestedDict_ListAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {'x':datastream_x,'y':datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data_x = self.stream['x'][self.pointer:self.pointer+1]
                data_y = self.stream['y'][self.pointer:self.pointer+1]
                self.pointer += 1
                self.send_output({'x':data_x,'y':data_y})

class NestedDict_NpArrayAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = {'x':datastream_x,'y':datastream_y}
        self.pointer = 0

    def agent_loop(self):
        if self.current_state == "Running":
            if self.pointer < num_samples:
                data_x = np.array(self.stream['x'][self.pointer:self.pointer+1])
                data_y = np.array(self.stream['y'][self.pointer:self.pointer+1])
                self.pointer += 1
                self.send_output({'x':data_x,'y':data_y})


params = [(SingleValueAgent, {'SingleValueAgent_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),
          (ListAgent, {'ListAgent_1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),
          (NpArrayAgent, {'NpArrayAgent_1': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}),
          (PdDataFrameAgent, {'PdDataFrameAgent_1': pd.DataFrame.from_dict({'x':datastream_x , 'y':datastream_y})}),
          (NestedDict_SingleValueAgent, {'NestedDict_SingleValueAgent_1': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]}}),
          (NestedDict_ListAgent, {'NestedDict_ListAgent_1': {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]}}),
          (NestedDict_NpArrayAgent, {'NestedDict_NpArrayAgent_1': {'x': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 'y': np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])}})
          ]

@pytest.mark.parametrize("agentType,expected_monitor_results", params)
def test_simpleAgent(agentType, expected_monitor_results):
    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=False)

    #init agents by adding into the agent network
    simple_agent = agentNetwork.add_agent(agentType=agentType)
    monitor_agent_1 = agentNetwork.add_agent(agentType=MonitorAgent)

    #shorten n wait loop time
    simple_agent.init_agent_loop(0.01)

    #connect agents
    agentNetwork.bind_agents(simple_agent, monitor_agent_1)

    # set all agents states to "Running"
    agentNetwork.set_running_state()
    time.sleep(test_timeout)

    # test to see if monitor agents have received the correct data
    assert str(monitor_agent_1.get_attr('memory')) == str(expected_monitor_results)

    # shutdown agent network
    agentNetwork.shutdown()
    time.sleep(3)
