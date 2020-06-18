"""
This example illustrates the use of agent pipeline and ML_Experiment to investigate the performance
of different pipelines with different hyperparameters.

We begin with a simple pipeline, of StandardScaler provided by sklearn and custom made Bayesian Neural Network model.
By wrapping them in the ML_Experiment object, their performances will be logged and saved into a folder ML_EXP
In the second page of the dashboard tabs, the results of each experiment,
consisting of pipelines and subsets called chains, can be viewed and compared.

"""


from agentMET4FOF.agents import AgentNetwork, MonitorAgent

from agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment
from agentMET4FOF_ml_extension.ML_Experiment import ML_Experiment
from agentMET4FOF_ml_extension.agents import AgentPipeline, ML_DataStreamAgent, ML_EvaluatorAgent
from agentMET4FOF_ml_extension.ml_uncertainty.bnn import BNN_Model
from agentMET4FOF_ml_extension.ml_uncertainty.evaluate_pred_unc import *

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def main():
    agentNetwork = AgentNetwork(dashboard_extensions=Dashboard_ML_Experiment)

    ml_exp_name = "simple"

    ML_Agent_pipelines_A = AgentPipeline(agentNetwork,
                                         [StandardScaler],
                                         [BNN_Model], hyperparameters=[[],
                                                                           [{"num_epochs":[500,1000],"task":["classification"],"architecture":[["d1","d1"],["d1","d1","d1"],["d1","d1","d1","d1"]]}]
                                                                           ])


    #init
    datastream_agent = agentNetwork.add_agent(agentType=ML_DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=ML_EvaluatorAgent)

    datastream_agent.init_parameters(data_name="iris", x=datasets.load_iris().data,y=datasets.load_iris().target)
    evaluation_agent.init_parameters([f1_score,p_acc_unc,avg_unc],[{"average":'micro'},{},{}], ML_exp=True)

    #setup ml experiment
    ml_experiment = ML_Experiment(agentNetwork, datasets=[datastream_agent], pipelines=[ML_Agent_pipelines_A], evaluation=[evaluation_agent], name=ml_exp_name, train_mode="Kfold5")

    #optional: connect evaluation agent to monitor agent
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    evaluation_agent.bind_output(monitor_agent)

    #set to active running
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
