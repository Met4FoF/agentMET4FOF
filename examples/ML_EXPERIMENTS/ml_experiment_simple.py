"""
This example illustrates the use of agent pipeline and ML_Experiment to investigate the performance
of different pipelines with different hyperparameters.

We begin with a simple pipeline, using the StandardScaler and GaussianProcessClassifier (GP) provided by sklearn.
By wrapping them in the ML_Experiment object, their performances will be logged and saved into a folder ML_EXP
In the second page of the dashboard tabs, the results of each experiment,
consisting of pipelines and subsets called chains, can be viewed and compared.

The example GP model reference derived from here:
https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html

"""


from agentMET4FOF.agents import AgentNetwork, MonitorAgent, AgentPipeline
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
from agentMET4FOF.ml_uncertainty.evaluate_pred_unc import p_acc_unc, avg_unc

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def main():
    agentNetwork = AgentNetwork()

    ml_exp_name = "simple"

    kernel_iso = 1.0 * RBF([1.0])
    kernel_ani = 1.0 * RBF([1.0]*4)

    ML_Agent_pipelines_A = AgentPipeline(agentNetwork,
                                         [StandardScaler],
                                         [GaussianProcessClassifier], hyperparameters=[[],
                                                                       [{"kernel":[kernel_iso,kernel_ani]}]
                                                                       ])


    #init
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=EvaluationAgent)

    datastream_agent.init_parameters(data_name="iris", x=datasets.load_iris().data,y=datasets.load_iris().target)
    evaluation_agent.init_parameters([f1_score],[{"average":'micro'}], ML_exp=True)

    #setup ml experiment
    ml_experiment = ML_Experiment(datasets=[datastream_agent], pipelines=[ML_Agent_pipelines_A], evaluation=[evaluation_agent], name=ml_exp_name, train_mode="Kfold5")
    agentNetwork.get_agent("Logger").set_ml_experiment(ml_experiment)

    #optional: connect evaluation agent to monitor agent
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    evaluation_agent.bind_output(monitor_agent)

    #set to active running
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
