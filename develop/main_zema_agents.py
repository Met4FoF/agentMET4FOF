from AgentMET4FOF import AgentNetwork, MonitorAgent, DataStreamAgent

from develop.develop_zema_datastream import ZEMA_DataStream
from develop.develop_zema_agents import TrainTestSplitAgent, FFT_BFCAgent, Pearson_FeatureSelectionAgent, LDA_Agent, Regression_Agent ,EvaluatorAgent
import numpy as np
np.random.seed(100)


if __name__ == '__main__':

    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    train_test_split_agent = agentNetwork.add_agent(agentType=TrainTestSplitAgent)
    fft_bfc_agent = agentNetwork.add_agent(agentType=FFT_BFCAgent)
    pearson_fs_agent = agentNetwork.add_agent(agentType=Pearson_FeatureSelectionAgent)
    lda_agent = agentNetwork.add_agent(agentType=LDA_Agent)
    bayesianRidge_agent = agentNetwork.add_agent(agentType=Regression_Agent, name="BayesianRidge_Agent")
    randomForest_agent = agentNetwork.add_agent(agentType=Regression_Agent, name="RandomForest_Agent")
    evaluator_agent = agentNetwork.add_agent(agentType=EvaluatorAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #init parameters
    datastream_agent.init_parameters(ZEMA_DataStream())
    train_test_split_agent.init_parameters(train_ratio=0.8)
    
    # init parameters for regression
    bayesianRidge_agent.init_parameters(regression_model="BayesianRidge")
    randomForest_agent.init_parameters(regression_model="RandomForest")

    #bind agents
    agentNetwork.bind_agents(datastream_agent, train_test_split_agent)
    agentNetwork.bind_agents(train_test_split_agent, fft_bfc_agent)
    agentNetwork.bind_agents(fft_bfc_agent, pearson_fs_agent)

    agentNetwork.bind_agents(pearson_fs_agent, lda_agent)
    agentNetwork.bind_agents(lda_agent, evaluator_agent)

    agentNetwork.bind_agents(pearson_fs_agent, bayesianRidge_agent)
    agentNetwork.bind_agents(bayesianRidge_agent, evaluator_agent)

    agentNetwork.bind_agents(pearson_fs_agent, randomForest_agent)
    agentNetwork.bind_agents(randomForest_agent, evaluator_agent)

    #bind to monitor agents
    agentNetwork.bind_agents(fft_bfc_agent, monitor_agent)
    agentNetwork.bind_agents(pearson_fs_agent, monitor_agent)

    agentNetwork.bind_agents(lda_agent, monitor_agent)
    agentNetwork.bind_agents(bayesianRidge_agent, monitor_agent)
    agentNetwork.bind_agents(randomForest_agent, monitor_agent)

    agentNetwork.bind_agents(evaluator_agent, monitor_agent)

    #trigger datastream to send all at once
    datastream_agent.send_all_sample()
