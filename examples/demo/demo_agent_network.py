from AgentMET4FOF import AgentNetwork

#Agent modules
import examples.ZEMA_EMC.zema_agents as zema_agents
import examples.ZEMA_EMC.zema_datastream as zema_datastream
import examples.demo.demo_agents as demo_agents

dashboard_modules = [zema_agents, zema_datastream, demo_agents]


if __name__ == '__main__':

    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=dashboard_modules)
