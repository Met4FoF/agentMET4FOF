from collections import deque
import numpy as np
from ..agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from ..streams import SineGenerator
import matplotlib.pyplot as plt
import numpy as np


class SensorAgent(AgentMET4FOF):
    def init_parameters(self, max_batch_size=10, max_buffer_size=100):
        self.buffer_time = deque(maxlen=max_buffer_size)
        self.buffer = deque(maxlen=max_buffer_size)

        self.max_buffer_size = max_buffer_size
        self.max_batch_size = max_batch_size

    def agent_loop(self):
        if self.current_state == "Running":

            # n_send sending operations are necessary, to fully reduce the buffer
            n_send = (len(self.buffer) - 1) // self.max_batch_size + 1

            for i in range(n_send):
                self.send_output({self.name: self.next_samples()})

    def next_samples(self):
        # take the next samples from the beginning of buffer
        n_pop = min(self.max_batch_size, len(self.buffer))
        next_samples = [(self.buffer_time.popleft(), self.buffer.popleft()) for i in range(n_pop)]

        return next_samples

    def append_to_buffer(self, timestamp, value):
        self.buffer_time.append(timestamp)
        self.buffer.append(value)


class SineGeneratorAgent_dict(AgentMET4FOF):
    def init_parameters(self, sensor_buffer_size=5):
        self.stream = SineGenerator()
        self.buffer_size = sensor_buffer_size

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #dictionary
            sine_data = {'x':sine_data['x'],'y':sine_data['x']+0.1}

            #save data into memory
            self.update_data_memory(self.name,sine_data)

            # send out buffered data if the stored data has exceeded the buffer size
            if self.buffer_filled:
                self.send_output(self.memory[self.name])
                self.memory = {}

class SineGeneratorAgent_deque(AgentMET4FOF):
    def init_parameters(self, max_batch_size=10,max_buffer_size=100):
        self.stream = SineGenerator()
        self.buffer = deque(maxlen=max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self.max_batch_size = max_batch_size

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #dictionary
            sine_data = {'x':sine_data['x'],'y':sine_data['x']+0.1}

            self.buffer.append(sine_data['x'])

            if len(self.buffer) >= self.max_batch_size:
                self.send_output({self.name: self.next_samples()})

    def next_samples(self):
        # take the next samples from the beginning of buffer
        n_pop = min(self.max_batch_size, len(self.buffer))
        next_samples = np.array([self.buffer.popleft() for i in range(n_pop)]).flatten()

        return next_samples

    def append_to_buffer(self, value):
        # self.buffer_time.append(timestamp)
        self.buffer.append(value)

if __name__ == "__main__":
    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=[],dashboard_update_interval=0.75,log_filename='log_name.csv')

    #init agents by adding into the agent network
    gen_dict_agent = agentNetwork.add_agent(agentType= SineGeneratorAgent_dict,log_mode=False)
    gen_deque_agent = agentNetwork.add_agent(agentType= SineGeneratorAgent_deque,log_mode=False)
    monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=5,log_mode=False)

    buffer_size = 10
    gen_dict_agent.init_parameters(sensor_buffer_size=buffer_size)
    gen_dict_agent.init_agent_loop(loop_wait=1)

    gen_deque_agent.init_parameters(max_batch_size=buffer_size)
    gen_deque_agent.init_agent_loop(loop_wait=1)

    #This monitor agent will only store 'x' of the data keys into its memory
    # monitor_agent.init_parameters(plot_filter=['x'])

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_dict_agent, monitor_agent)
    agentNetwork.bind_agents(gen_deque_agent, monitor_agent)


    # set all agents states to "Running"
    agentNetwork.set_running_state()

