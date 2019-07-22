
import numpy as np

from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStreamAgent

# -*- coding: utf-8 -*-
class Addition(AgentMET4FOF):
    def init_parameters(self, addBy=100):
        self.addBy = addBy

    def on_received_message(self, message):
        proc_data = message['data'] + self.addBy
        return self.send_output(proc_data)

class Subtract(AgentMET4FOF):
    def init_parameters(self, subtractBy=100):
        self.subtractBy = subtractBy

    def on_received_message(self, message):
        proc_data = message['data'] - self.subtractBy
        return self.send_output(proc_data)

class Sensor(AgentMET4FOF):

    def on_init(self):
        super().on_init()

        self.DataSource = np.arange(0, 1000)
        self.read_index = 0
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
        self.current_state = self.states[0]
        self.start_agent_loop(n_wait=1.0)

    def agent_loop(self):
        if self.current_state == self.states[1]:
            data = self.read_data()
            if data is not None:
                self.send_output(data)
        else:
            return 0

    def read_data(self):
        if self.read_index < len(self.DataSource):
            #get data from current index
            data = self.DataSource[self.read_index]

            #advance index
            self.read_index+=1
        else:
            data = None

        #log
        self.log_info(data)
        return data


if __name__ == '__main__':
    #start agent network
    agentNetwork = AgentNetwork()

    #init agents
    data_stream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    ml_agent = agentNetwork.add_agent(agentType=ML_Model)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #connect agents
    agentNetwork.bind_agents(data_stream_agent, ml_agent)
    agentNetwork.bind_agents(ml_agent, monitor_agent)

    data_stream_agent.bind_output(ml_agent)
    ml_agent.bind_output(monitor_agent)


    #test looping
    data_stream_agent.init_agent_loop(3.0)

    #agentNetwork.set_running_state()
    #agentNetwork.shutdown()



