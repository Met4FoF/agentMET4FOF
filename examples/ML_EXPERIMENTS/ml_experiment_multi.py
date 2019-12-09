"""
In this example, we extend the ML_Experiment to include two simultaneuous pipelines instead of one.
Particularly, in the first pipeline, we have only the BNN model which implies direct mapping from
the raw data to the evaluation agent, while the second pipeline has a StandardScaler before the
BNN model.

"""


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

    ml_exp_name = "multiple"

    ML_Agent_pipelines_A = AgentPipeline(agentNetwork,
                                             [BNN_Model], hyperparameters=[
                                                                           [{"num_epochs":[500],"task":["classification"],"architecture":[["d1","d1"],["d1","d1","d1"],["d1","d1","d1","d1"]]}]
                                                                           ])
    ML_Agent_pipelines_B = AgentPipeline(agentNetwork,
                                             [StandardScaler],
                                             [BNN_Model], hyperparameters=[[],
                                                                           [{"num_epochs":[500],"task":["classification"],"architecture":[["d1","d1"],["d1","d1","d1"],["d1","d1","d1","d1"]]}]
                                                                           ])


    #init
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=EvaluationAgent)

    datastream_agent.init_parameters(data_name="iris", x=datasets.load_iris().data,y=datasets.load_iris().target)
    evaluation_agent.init_parameters([f1_score,p_acc_unc,avg_unc],[{"average":'micro'},{},{}], ML_exp=True)

    #setup ml experiment
    ml_experiment = ML_Experiment(datasets=[datastream_agent], pipelines=[ML_Agent_pipelines_A, ML_Agent_pipelines_B], evaluation=[evaluation_agent], name=ml_exp_name, train_mode="Kfold5")
    agentNetwork.get_agent("Logger").set_ml_experiment(ml_experiment)

    #optional: connect evaluation agent to monitor agent
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    evaluation_agent.bind_output(monitor_agent)

    #set to active running
    agentNetwork.set_running_state()
