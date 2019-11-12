from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

#The function of buffering memory in MonitorAgent and SensorAgent (represented by SineGenerator)
#is displayed here
#For the SensorAgent, the sensor buffer size is defined in init_parameters in an ad-hoc manner
#as it is not built into the agent framework yet
#Whereas for MonitorAgent, which leverages on the built-in update_data_memory
#the memory_buffer_size is specified upon agent instantiation

import matplotlib.pyplot as plt
import numpy as np

class SineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self, sensor_buffer_size=5):
        self.stream = SineGenerator()
        self.buffer_size = sensor_buffer_size

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #dictionary
            sine_data = {'x':sine_data['x'],'y':sine_data['x']+0.1}
            
            #save data into memory
            self.update_data_memory({'from':self.name,'data':sine_data})
            # send out buffered data if the stored data has exceeded the buffer size
            if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                self.send_output(self.memory[self.name])
                self.memory = {}

def main():
    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=[],dashboard_update_interval=0.75,log_filename='log_name.csv')

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType= SineGeneratorAgent,log_mode=False)
    monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=5,log_mode=False)
    monitor_agent_2 = agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=10,log_mode=False)

    gen_agent.init_parameters(sensor_buffer_size=1)
    gen_agent.init_agent_loop(loop_wait=1)

    #This monitor agent will only store 'x' of the data keys into its memory
    monitor_agent.init_parameters(plot_filter=['x'])

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output()
    gen_agent.bind_output(monitor_agent)
    gen_agent.bind_output(monitor_agent_2)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()

