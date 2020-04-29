from agentMET4FOF.agents import AgentNetwork

#Agent modules
import examples.demo.demo_agents as demo_agents

dashboard_modules = [demo_agents]


def main():
    # start agent network server and return it to allow for shutdown
    return AgentNetwork(dashboard_modules=dashboard_modules)


if __name__ == '__main__':
    main()
