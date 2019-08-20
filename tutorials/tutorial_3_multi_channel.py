from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator, CosineGenerator

#We can use different channels for the receiver  to handle specifically each channel name
#This can be useful for example in splitting train and test channels in machine learning
#Then, the user will need to implement specific handling of each channel in the receiving agent
#In this example, the MultiGeneratorAgent is used to send two different types of data - Sine and Cosine generator
#This is done via specifiying send_output(channel="sine") and send_output(channel="cosine")
#Then on the receiving end, the on_received_message() function checks for message['channel'] to handle it separately
#Note that by default, Monitor Agent is only subscribed to the "default" channel
#Hence it will not respond to the "cosine" and "sine" channel

def minus(data, minus_val):
    return data-minus_val

def plus(data,plus_val):
    return data+plus_val

class MultiGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.sine_stream = SineGenerator()
        self.cos_stream = CosineGenerator()

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.sine_stream.next_sample() #dictionary
            cosine_data = self.sine_stream.next_sample() #dictionary
            self.send_output(sine_data['x'], channel="sine")
            self.send_output(cosine_data['x'], channel="cosine")

class MultiOutputMathAgent(AgentMET4FOF):
    def init_parameters(self,minus_param=0.5,plus_param=0.5):
        self.minus_param = minus_param
        self.plus_param = plus_param

    def on_received_message(self, message):
        """
        Checks for message['channel'] and handles them separately
        Acceptable channels are "cosine" and "sine"
        """
        if message['channel'] == "cosine":
            minus_data = minus(message['data'], self.minus_param)
            self.send_output({'cosine_minus':minus_data})
        elif message['channel'] == 'sine':
            plus_data = plus(message['data'], self.plus_param)
            self.send_output({'sine_plus':plus_data})


def main():
    # start agent network server
    agentNetwork = AgentNetwork()
    # init agents
    gen_agent = agentNetwork.add_agent(agentType=MultiGeneratorAgent)
    multi_math_agent = agentNetwork.add_agent(agentType=MultiOutputMathAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    # connect agents : We can connect multiple agents to any particular agent
    # However the agent needs to implement handling multiple inputs
    agentNetwork.bind_agents(gen_agent, multi_math_agent)
    agentNetwork.bind_agents(gen_agent, monitor_agent)
    agentNetwork.bind_agents(multi_math_agent, monitor_agent)
    # set all agents states to "Running"
    agentNetwork.set_running_state()


    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()



