from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

#Here we define a new agent SineGeneratorAgent, and override the functions : init_parameters & agent_loop
#init_parameters() is used to setup the data stream and necessary parameters
#agent_loop() is an infinite loop, and will read from the stream continuously,
#then it sends the data to its output channel via send_output
#Each agent has internal current_state which can be used as a switch by the AgentNetwork
import torch
class Cuda_Agent(AgentMET4FOF):
    def init_parameters(self):
        y = torch.tensor([1., 2.]).cuda()

def main():
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType= Cuda_Agent)
    monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent)

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output()
    gen_agent.bind_output(monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()

