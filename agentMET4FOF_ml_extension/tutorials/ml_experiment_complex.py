"""
A more complicated pipeline is experimented here, with a larger range of hyperparameters.
Since this implies a grid search, the computation can be too intensive if the pipeline and
hyperparameter range are made too large.

"""



from agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment
from agentMET4FOF_ml_extension.datastream import *
from agentMET4FOF_ml_extension.evaluator import *
from agentMET4FOF_ml_extension.ML_Experiment import *
from agentMET4FOF_ml_extension.agents import AgentPipeline

from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,PowerTransformer


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from agentMET4FOF_ml_extension.ml_uncertainty.evaluate_pred_unc import *

def main():
    #initialise with dashboard ml experiments
    agentNetwork = AgentNetwork(dashboard_extensions=Dashboard_ML_Experiment)

    ml_exp_name = "complex"

    kernel_iso = 1.0 * RBF([1.0])

    ML_Agent_pipelines_A = AgentPipeline(agentNetwork,
                                         [StandardScaler, RobustScaler,MinMaxScaler,MaxAbsScaler,PowerTransformer],
                                         [PCA],
                                         [GaussianProcessClassifier], hyperparameters=[[],[],[{"kernel":[kernel_iso]}]])


    #init
    datastream_agent = agentNetwork.add_agent(agentType=ML_DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=ML_EvaluatorAgent)

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
