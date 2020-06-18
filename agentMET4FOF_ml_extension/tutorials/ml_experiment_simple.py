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


from agentMET4FOF.agents import AgentNetwork, MonitorAgent

from agentMET4FOF_ml_extension.ML_Experiment import ML_Experiment
from agentMET4FOF_ml_extension.agents import AgentPipeline, ML_DataStreamAgent, ML_EvaluatorAgent
from agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment

from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def main():
    #initialise with dashboard ml experiments
    agentNetwork = AgentNetwork(dashboard_extensions=Dashboard_ML_Experiment)

    ml_exp_name = "simple"

    kernel_iso = 1.0 * RBF([1.0])
    kernel_ani = 1.0 * RBF([1.0]*4)

    ML_Agent_pipelines_A = AgentPipeline(agentNetwork,
                                         [StandardScaler],
                                         [GaussianProcessClassifier], hyperparameters=[[],
                                                                       [{"kernel":[kernel_iso,kernel_ani]}]
                                                                       ])


    #init
    datastream_agent = agentNetwork.add_agent(agentType=ML_DataStreamAgent)
    evaluation_agent = agentNetwork.add_agent(agentType=ML_EvaluatorAgent)

    datastream_agent.init_parameters(data_name="iris", x=datasets.load_iris().data,y=datasets.load_iris().target)
    evaluation_agent.init_parameters([f1_score],[{"average":'micro'}], ML_exp=True)

    #setup ml experiment
    ml_experiment = ML_Experiment(agentNetwork, datasets=[datastream_agent],
                                  pipelines=[ML_Agent_pipelines_A],
                                  evaluation=[evaluation_agent],
                                  name=ml_exp_name,
                                  train_mode="Kfold5")

    #optional: connect evaluation agent to monitor agent
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    evaluation_agent.bind_output(monitor_agent)

    #set to active running
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
