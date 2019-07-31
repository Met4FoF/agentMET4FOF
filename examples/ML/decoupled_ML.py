#AGENTMET4FOF modules
from agentMet4FoF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent, \
    DataStreamAgent
from agentMet4FoF.streams import extract_x_y

#ML dependencies
import numpy as np
from skmultiflow.data import SineGenerator
from skmultiflow.trees import HoeffdingTree


#Here we demonstrate a slightly more complicated scenario :
#We separate the ML into different agents: Predictor, Trainer and Evaluator
#This has the advantage in actual hardware deployment where Predictor Agent is most likely
#to be deployed on limited computing resource compared to Trainer Agent which may be more resourceful
#However, it is important to note that - the predictor and trainer has not been synchronized
#upon receiving the data from the generator!
class Predictor(AgentMET4FOF):
    def init_parameters(self, ml_model= None):
        self.ml_model =ml_model

    def on_received_message(self, message):

        #handle Trainer Agent
        if message['senderType'] == "Trainer":
            try:
                data = message['data']
                if 'ml_model' in data.keys():
                    self.ml_model = data['ml_model']
                    return 0
            except:
                raise Exception

        #handle x & y data
        else:
            if self.ml_model is not None:
                x , y_true = extract_x_y(message)
                y_pred = self.ml_model.predict(x)
                self.send_output({'y_pred':y_pred, 'y_true':y_true})

class Trainer(AgentMET4FOF):
    def init_parameters(self, ml_model= HoeffdingTree()):
        self.ml_model = ml_model

    def on_received_message(self, message):
        x,y = extract_x_y(message)
        self.ml_model.partial_fit(x, y)
        self.send_output({'ml_model':self.ml_model})

class Evaluator(AgentMET4FOF):
    # classifier accuracy - user defined
    def compute_accuracy(self, y_pred, y_true):
        res = y_pred == y_true
        num_accurate = [1 if y == True else 0 for y in res]
        accuracy = np.sum(num_accurate) / len(num_accurate) * 100
        return accuracy

    def on_received_message(self, message):
        if message['senderType'] == "Predictor":
            data = message['data']
            y_pred = data['y_pred']
            y_true = data['y_true']
            res = self.compute_accuracy(y_pred=y_pred, y_true=y_true)
            self.send_output(res)


def main():
    # start agent network server
    agentNetwork = AgentNetwork()
    # init agents
    gen_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    trainer_agent = agentNetwork.add_agent(agentType=Trainer)
    predictor_agent = agentNetwork.add_agent(agentType=Predictor)
    evaluator_agent = agentNetwork.add_agent(agentType=Evaluator)
    monitor_agent_1 = agentNetwork.add_agent(agentType=MonitorAgent)
    monitor_agent_2 = agentNetwork.add_agent(agentType=MonitorAgent)
    gen_agent.init_parameters(stream=SineGenerator(), pretrain_size=1000,
                              batch_size=1)
    trainer_agent.init_parameters(ml_model=HoeffdingTree())
    # connect agents : We can connect multiple agents to any particular agent
    # However the agent needs to implement handling multiple input types
    agentNetwork.bind_agents(gen_agent, trainer_agent)
    agentNetwork.bind_agents(gen_agent, predictor_agent)
    agentNetwork.bind_agents(trainer_agent, predictor_agent)
    agentNetwork.bind_agents(predictor_agent, evaluator_agent)
    agentNetwork.bind_agents(evaluator_agent, monitor_agent_1)
    agentNetwork.bind_agents(predictor_agent, monitor_agent_2)
    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
