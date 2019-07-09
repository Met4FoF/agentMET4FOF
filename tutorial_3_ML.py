
import numpy as np

from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStream

#We demonstrate the use of pre-made agents for machine learning : DataStream agent and ML_Model agent
#The agents are compatible with scikit-multiflow package
#Here we demonstrate the implementation of multi-data streams and multi machine learning models in parallel

from skmultiflow.data import WaveformGenerator, SineGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.bayes import NaiveBayes

if __name__ == '__main__':

    #start agent network
    agentNetwork = AgentNetwork()
    agentNetwork.start_server()

    #init agents
    data_stream_agent_1 = agentNetwork.add_agent(agentType=DataStream)
    ml_agent_hoeffdingTree = agentNetwork.add_agent(agentType=ML_Model)
    ml_agent_neuralNets = agentNetwork.add_agent(agentType=ML_Model)
    monitor_agent_1 = agentNetwork.add_agent(agentType=MonitorAgent)

    #init parameters
    data_stream_agent_1.init_parameters(stream=WaveformGenerator(), pretrain_size=100, max_samples=100000, batch_size=100)
    ml_agent_hoeffdingTree.init_parameters(ml_model= HoeffdingTree())
    ml_agent_neuralNets.init_parameters(ml_model=NaiveBayes())

    #connect agents
    agentNetwork.bind_agents(data_stream_agent_1, ml_agent_hoeffdingTree)
    agentNetwork.bind_agents(data_stream_agent_1, ml_agent_neuralNets)
    agentNetwork.bind_agents(ml_agent_hoeffdingTree, monitor_agent_1)
    agentNetwork.bind_agents(ml_agent_neuralNets, monitor_agent_1)

    agentNetwork.set_running_state()



