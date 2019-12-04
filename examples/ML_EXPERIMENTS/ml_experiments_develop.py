# import os
# path = "F:/PhD Research/Github/develop_ml_experiments_met4fof/agentMET4FOF"
# os.chdir(path)

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent, AgentPipeline
from agentMET4FOF.develop.datastream import *
from agentMET4FOF.develop.evaluator import *
from agentMET4FOF.develop.ML_Experiment import *

from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from pprint import pprint
import copy

if __name__ == "__main__":
    agentNetwork = AgentNetwork()
    hyperparameters = [[],
                      [{"n_components":[1,2,3]}],
                      ]


    # ML_Agent_pipelines_B = AgentPipeline(agentNetwork,
    #                                      [StandardScaler,RobustScaler],
    #                                      [PCA],
    #                                      [LogisticRegression,SVC],
    #                                      hyperparameters=hyperparameters)

    ML_Agent_pipelines_B = AgentPipeline(agentNetwork,
                                             [StandardScaler],
                                             [LogisticRegression])

    #init
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=EvaluationAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    datastream_agent.init_parameters(datasets.load_iris())
    evaluation_agent.init_parameters(f1_score,average='micro', ML_exp=True)

    #binding
    evaluation_agent.bind_output(monitor_agent)
    ML_Agent_pipelines_B.bind_output(evaluation_agent)
    datastream_agent.bind_output(ML_Agent_pipelines_B)

    ml_experiment = ML_Experiment(pipelines=[ML_Agent_pipelines_B])
    agentNetwork.get_agent("Logger").set_ml_experiment(ml_experiment)
    print(ml_experiment.pipeline_details)
    print(ml_experiment.chain_results)
    agentNetwork.set_running_state()

    #save and load file
    save_experiment(agentNetwork.get_agent("Logger").get_attr("ml_experiment"))
    ml_exp_load = load_experiment("run_2")
