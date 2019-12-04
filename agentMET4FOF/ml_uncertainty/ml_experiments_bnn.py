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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,PowerTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from pprint import pprint
import copy

from agentMET4FOF.ml_uncertainty.bnn import BNN_Model
from agentMET4FOF.ml_uncertainty.evaluate_pred_unc import *

if __name__ == "__main__":
    agentNetwork = AgentNetwork()
    hyperparameters = [[],
                      [{"n_components":[1,2,3]}],
                      ]


    # ML_Agent_pipelines_B = AgentPipeline(agentNetwork,
    #                                      [StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler,PowerTransformer],
    #                                      [PCA],
    #                                      [LogisticRegression,SVC],
    #                                      hyperparameters=hyperparameters)

    # ML_Agent_pipelines_B = AgentPipeline(agentNetwork,
    #                                          [StandardScaler, RobustScaler,MinMaxScaler,MaxAbsScaler,PowerTransformer],
    #                                          [LogisticRegression])

    # ML_Agent_pipelines_B = AgentPipeline(agentNetwork,
    #                                          [StandardScaler, RobustScaler,MinMaxScaler,MaxAbsScaler,PowerTransformer],
    #                                          [BNN_Model], hyperparameters=[[],
    #                                                                        [{}, {"num_epochs":[500,1000,1500],"return_unc":[False],"task":["classification"],"architecture":[["d1","d1"],["d1","d1","d1","d1"]]}]
    #                                                                        ])

    ML_Agent_pipelines_B = AgentPipeline(agentNetwork,
                                             [StandardScaler],
                                             [BNN_Model], hyperparameters=[[],
                                                                           [{"num_epochs":[500,1000,1500],"task":["classification"],"architecture":[["d1","d1"],["d1","d1","d1","d1"]]}]
                                                                           ])
    #init
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=EvaluationAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    datastream_agent.init_parameters(datasets.load_iris())
    # evaluation_agent.init_parameters(f1_score,average='micro', ML_exp=True)
    evaluation_agent.init_parameters([f1_score,p_acc_unc,avg_unc],[{"average":'micro'},{},{}], ML_exp=True)
    #binding
    evaluation_agent.bind_output(monitor_agent)
    ML_Agent_pipelines_B.bind_output(evaluation_agent)
    datastream_agent.bind_output(ML_Agent_pipelines_B)
    datastream_agent.bind_output(ML_Agent_pipelines_B.pipeline[1])

    #setup ml experiment
    ml_experiment = ML_Experiment(pipelines=[ML_Agent_pipelines_B],name="compare_effect_scalers")
    agentNetwork.get_agent("Logger").set_ml_experiment(ml_experiment)
    print(ml_experiment.pipeline_details)
    print(ml_experiment.chain_results)

    #set to active running
    agentNetwork.set_running_state()

    #save and load file
    save_experiment(agentNetwork.get_agent("Logger").get_attr("ml_experiment"))
    ml_exp_load = load_experiment("run_2")
