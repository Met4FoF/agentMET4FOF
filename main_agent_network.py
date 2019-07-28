from AgentMET4FOF import AgentNetwork

#Agent modules
import develop.develop_zema_agents as zema_agents
import develop.develop_zema_datastream as zema_datastream

dashboard_modules = [zema_agents, zema_datastream]

if __name__ == '__main__':

    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=dashboard_modules)
