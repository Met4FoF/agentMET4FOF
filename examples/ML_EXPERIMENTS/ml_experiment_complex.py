"""
A more complicated pipeline is experimented here, with a larger range of hyperparameters.
Since this implies a grid search, the computation can be too intensive if the pipeline and
hyperparameter range are made too large.
"""



from agentMET4FOF.agents import AgentNetwork, MonitorAgent, AgentPipeline
from agentMET4FOF.develop.datastream import *
from agentMET4FOF.develop.evaluator import *
from agentMET4FOF.develop.ML_Experiment import *

from sklearn import datasets

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,PowerTransformer

from examples.ML_EXPERIMENTS.bnn import BNN_Model
from examples.ML_EXPERIMENTS.evaluate_pred_unc import *

def main():
    agentNetwork = AgentNetwork()

    ml_exp_name = "complex"

    ML_Agent_pipelines_A = AgentPipeline(agentNetwork,
                                         [StandardScaler, RobustScaler,MinMaxScaler,MaxAbsScaler,PowerTransformer],
                                         [PCA],
                                         [BNN_Model], hyperparameters=[[],[],[{"num_epochs":[500,1000,1500],
                                                                               "task":["classification"],
                                                                               "architecture":[["d1","d1"],["d1","d1","d1"]]}]])


    #init
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=EvaluationAgent)

    datastream_agent.init_parameters(data_name="iris", x=datasets.load_iris().data,y=datasets.load_iris().target)
    evaluation_agent.init_parameters([f1_score,p_acc_unc,avg_unc],[{"average":'micro'},{},{}], ML_exp=True)

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
