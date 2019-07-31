from agentMet4FoF.agents import AgentNetwork

#Agent modules
import examples.ZEMA_EMC.zema_agents as zema_agents
import examples.ZEMA_EMC.zema_datastream as zema_datastream
import examples.demo.demo_agents as demo_agents

dashboard_modules = [zema_agents, zema_datastream, demo_agents]


def main():
    # start agent network server and return it to allow for shutdown
    return AgentNetwork(dashboard_modules=dashboard_modules)


if __name__ == '__main__':
    main()
