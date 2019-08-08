from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent

def minus(data, minus_val):
    return data-minus_val

def plus(data,plus_val):
    return data+plus_val

class SubtractAgent(AgentMET4FOF):
    def init_parameters(self,minus_param=0.5):
        self.minus_param = minus_param

    def on_received_message(self, message):
        minus_data = minus(message['data']['x'], self.minus_param)

        self.send_output(minus_data)

class AdditionAgent(AgentMET4FOF):
    def init_parameters(self,plus_param=0.5):
        self.plus_param = plus_param

    def on_received_message(self, message):
        plus_data = plus(message['data']['x'], self.plus_param)

        self.send_output(plus_data)


def main():
    # start agent network server
    agentNetwork = AgentNetwork()
    # init agents
    sub_agent = agentNetwork.add_agent(agentType=SubtractAgent)
    add_agent = agentNetwork.add_agent(agentType=AdditionAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    # connect
    agentNetwork.bind_agents(sub_agent, monitor_agent)
    agentNetwork.bind_agents(add_agent, monitor_agent)
    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
