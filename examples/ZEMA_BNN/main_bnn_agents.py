from agentMET4FOF.agents import AgentNetwork,AgentMET4FOF, MonitorAgent, DataStreamAgent
from examples.ZEMA_BNN.zema_hyd_datastream import ZEMA_Hyd_DataStream
from examples.ZEMA_BNN.bnn_agents import StatsFeaturesAgent, BNN_Agent, BNN_Model, EvaluatorAgent
from examples.ZEMA_EMC.zema_agents import TrainTestSplitAgent, FFT_BFCAgent, Pearson_FeatureSelectionAgent, LDA_Agent, Regression_Agent

import torch.multiprocessing as mp
import numpy as np

np.random.seed(100)

USE_CUDA = False

#output_sizes = [3,4,3,4,2]

def main():
    #if use cuda
    if USE_CUDA:
        mp.set_start_method('spawn',force=True)


    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)

    train_test_split_agent = agentNetwork.add_agent(agentType=TrainTestSplitAgent)
    stats_features_agent = agentNetwork.add_agent(agentType=StatsFeaturesAgent)

    bnn_agent = agentNetwork.add_agent(agentType=BNN_Agent)
    evaluator_agent = agentNetwork.add_agent(agentType=EvaluatorAgent)

    monitor_agent_1 = agentNetwork.add_agent(agentType=MonitorAgent)
    monitor_agent_2 = agentNetwork.add_agent(agentType=MonitorAgent)

    #init parameters
    datastream_agent.init_parameters(stream=ZEMA_Hyd_DataStream(),pretrain_size=-1, randomize=True)
    train_test_split_agent.init_parameters(train_ratio=0.8)
    bnn_agent.init_parameters(model=BNN_Model, output_size=4, selectY_col = 3)

    #connect agents by either way:
    agentNetwork.bind_agents(datastream_agent, train_test_split_agent)
    agentNetwork.bind_agents(train_test_split_agent,stats_features_agent)
    #agentNetwork.bind_agents(train_test_split_agent,bnn_agent)
    agentNetwork.bind_agents(stats_features_agent,bnn_agent)
    agentNetwork.bind_agents(bnn_agent,monitor_agent_1)
    agentNetwork.bind_agents(bnn_agent,evaluator_agent)
    agentNetwork.bind_agents(evaluator_agent,monitor_agent_2)


    # # set all agents states to "Running"
    agentNetwork.set_running_state()


if __name__ == '__main__':
    main()
