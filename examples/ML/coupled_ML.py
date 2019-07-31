#AGENTMET4FOF modules
from agentMet4FoF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent, \
    DataStreamAgent
from agentMet4FoF.streams import extract_x_y

#ML Dependencies
from skmultiflow.data import WaveformGenerator
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTree
from sklearn.model_selection import StratifiedKFold
import numpy as np
#We demonstrate the use of pre-made agents for machine learning : DataStream agent and ML_Model agent
#The agents are compatible with scikit-multiflow package
#Here we demonstrate the implementation of multi-data streams and multi machine learning models in parallel

class ML_Model(AgentMET4FOF):
    def init_parameters(self, mode="prequential", ml_model= HoeffdingTree(), split_type=None):
        self.mode = mode
        self.ml_model = ml_model
        self.results = []
        if split_type is not None:
            self.split_type = split_type
        else:
            self.split_type = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def on_received_message(self, message):
        #handle data structure to extract features & target
        try:
            x,y = extract_x_y(message)
        except:
            raise Exception

        # prequential: test & train
        if self.mode == "prequential":
            y_pred = self.ml_model.predict(x)
            self.ml_model.partial_fit(x, y)
            res = self.compute_accuracy(y_pred=y_pred, y_true=y)
            self.results.append(res)

        # holdout: test & train
        elif self.mode == "holdout":
            res_temp = []
            # begin kfold
            for train_index, test_index in self.split_type.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.ml_model.partial_fit(x_train, y_train)
                y_pred = self.ml_model.predict(x_test)
                res = self.compute_accuracy(y_pred=y_pred, y_true=y_test)
                res_temp.append(res)
            self.results.append(np.mean(res_temp))

        self.send_output(self.results[-1])

    # classifier accuracy - user defined
    def compute_accuracy(self, y_pred, y_true):
        res = y_pred == y_true
        num_accurate = [1 if y == True else 0 for y in res]
        accuracy = np.sum(num_accurate) / len(num_accurate) * 100
        return accuracy


def main():
    global agentNetwork
    # start agent network
    agentNetwork = AgentNetwork()
    # add agents
    data_stream_agent_1 = agentNetwork.add_agent(agentType=DataStreamAgent)
    ml_agent_hoeffdingTree = agentNetwork.add_agent(agentType=ML_Model)
    ml_agent_neuralNets = agentNetwork.add_agent(agentType=ML_Model)
    monitor_agent_1 = agentNetwork.add_agent(agentType=MonitorAgent)
    # init parameters
    data_stream_agent_1.init_parameters(stream=WaveformGenerator(),
                                        pretrain_size=1000, batch_size=100)
    ml_agent_hoeffdingTree.init_parameters(ml_model=HoeffdingTree())
    ml_agent_neuralNets.init_parameters(ml_model=NaiveBayes())
    # connect agents
    agentNetwork.bind_agents(data_stream_agent_1, ml_agent_hoeffdingTree)
    agentNetwork.bind_agents(data_stream_agent_1, ml_agent_neuralNets)
    agentNetwork.bind_agents(ml_agent_hoeffdingTree, monitor_agent_1)
    agentNetwork.bind_agents(ml_agent_neuralNets, monitor_agent_1)
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()




