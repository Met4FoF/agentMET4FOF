from agentMET4FOF.agents import AgentNetwork, MonitorAgent, DataStreamAgent


from examples.ZEMA_EMC.zema_agents import TrainTestSplitAgent, FFT_BFCAgent, Pearson_FeatureSelectionAgent, LDA_Agent, EvaluatorAgent
from examples.ZEMA_EMC.zema_feature_extract import Pearson_FeatureSelection, FFT_BFC
from examples.ZEMA_EMC.zema_datastream import ZEMA_DataStream
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import time

timeout_wait = 60


def test_zema_emc_lda():
    np.random.seed(100)

    #prepare components
    zema_datastream = ZEMA_DataStream()
    fft_bfc = FFT_BFC()
    pearson_fs = Pearson_FeatureSelection()
    ml_model = LinearDiscriminantAnalysis(n_components=3,priors=None, shrinkage=None, solver='eigen')

    def reformat_target(target_vector):
        class_target_vector=np.ceil(target_vector[0])
        for i in class_target_vector.index:
            if class_target_vector[i]== 0:
                class_target_vector[i]= 1                   #Fixing the zero element.
        return np.array(class_target_vector)

    #get data
    zema_data = zema_datastream.all_samples()

    #split
    x_data = zema_data['x']
    y_data = zema_data['y']

    x_train, x_test =train_test_split(x_data, train_size=0.8,random_state=15)
    y_train, y_test =train_test_split(y_data, train_size=0.8,random_state=15)

    #train
    x_train = fft_bfc.fit_transform(x_train)
    x_train, sensor_perc_train = pearson_fs.fit_transform(x_train, y_train)

    y_train = reformat_target(y_train)
    ml_model = ml_model.fit(x_train, y_train)
    print("Overall Train Score: " + str(ml_model.score(x_train, y_train)))

    #test
    x_test = fft_bfc.transform(x_test)
    x_test, sensor_perc_test = pearson_fs.transform(x_test)

    y_test = reformat_target(y_test)
    print("Overall Test Score: " + str(ml_model.score(x_test, y_test)))

    assert ml_model.score(x_test, y_test) == 0.8204924543288324


def test_zema_emc_lda_agents():
    np.random.seed(100)
    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=False)

    #init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    train_test_split_agent = agentNetwork.add_agent(agentType=TrainTestSplitAgent)
    fft_bfc_agent = agentNetwork.add_agent(agentType=FFT_BFCAgent)
    pearson_fs_agent = agentNetwork.add_agent(agentType=Pearson_FeatureSelectionAgent)
    lda_agent = agentNetwork.add_agent(agentType=LDA_Agent)
    evaluator_agent = agentNetwork.add_agent(agentType=EvaluatorAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #init parameters
    datastream_agent.init_parameters(ZEMA_DataStream())
    train_test_split_agent.init_parameters(train_ratio=0.8)

    #bind agents
    agentNetwork.bind_agents(datastream_agent, train_test_split_agent)
    agentNetwork.bind_agents(train_test_split_agent, fft_bfc_agent)
    agentNetwork.bind_agents(fft_bfc_agent, pearson_fs_agent)
    agentNetwork.bind_agents(pearson_fs_agent, lda_agent)
    agentNetwork.bind_agents(lda_agent, evaluator_agent)

    #bind to monitor agents
    agentNetwork.bind_agents(fft_bfc_agent, monitor_agent)
    agentNetwork.bind_agents(pearson_fs_agent, monitor_agent)
    agentNetwork.bind_agents(lda_agent, monitor_agent)
    agentNetwork.bind_agents(evaluator_agent, monitor_agent)

    #trigger datastream to send all at once
    datastream_agent.send_all_sample()

    time.sleep(timeout_wait)

    assert lda_agent.get_attr('lda_test_score') == 0.8204924543288324

    agentNetwork.shutdown()
