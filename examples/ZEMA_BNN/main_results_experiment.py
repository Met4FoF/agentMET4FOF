from agentMET4FOF.agents import AgentNetwork
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#investigate uncertainty & accuracy relationship
def get_p_accuracy_certainty(y_res,uncertainty_threshold=10):
    """
    Parameters
    ----------
    y_res : DataFrame
        Dataframe with following columns : `accurate` and `certainty`
        where y_res.accurate is a Series of boolean,
        y_res.certainty is a Series of floats

    uncertainty_threshold : float
        Predictions with `certainty` which is below the
        threshold will be declared uncertain

    Returns
    -------
    p_acc_certain, p_acc_uncertain: float, float
        Probability that a prediction is accurate, given it is certain
        and probability that a prediction is accurate, given it is uncertain

    p_certain : float
        Percentage of certain cases
    p_uncertain : float
        1-p_certain will be percentage of uncertain cases
    """

    uncertain_y_res = y_res[y_res['certainty'] < uncertainty_threshold]
    certain_y_res = y_res[y_res['certainty'] >= uncertainty_threshold]
    try:
        p_acc_certain = 100*certain_y_res.accurate.value_counts()[True]/certain_y_res.accurate.count()
        p_acc_uncertain = 100*uncertain_y_res.accurate.value_counts()[True]/uncertain_y_res.accurate.count()

        # p_acc_uncertain =0
        return p_acc_certain,p_acc_uncertain
    except Exception as e:
        print("ERROR OCCURED ", e)
        print(uncertainty_threshold)
        print(certain_y_res.accurate.value_counts())
        print(uncertain_y_res.accurate.value_counts())
        return -1,-1

def get_p_certain(y_res,uncertainty_threshold=10):
    uncertain_y_res = y_res[y_res['certainty'] < uncertainty_threshold]
    certain_y_res = y_res[y_res['certainty'] >= uncertainty_threshold]

    try:

        p_certain =100*certain_y_res.certain.count()/y_res.shape[0]
        p_uncertain =100*uncertain_y_res.certain.count()/y_res.shape[0]
        return p_certain, p_uncertain
    except Exception as e:
        print("ERROR OCCURED ", e)
        print(uncertainty_threshold)
        print(certain_y_res.accurate.value_counts())
        print(uncertain_y_res.accurate.value_counts())
        return p_certain, p_uncertain

def plot_subplots_acc_certainty(p_acc_certain_list,p_acc_uncertain_list,p_certain_list,p_uncertain_list,save_file=None):
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,15))
    ax1.plot(p_acc_certain_list)
    ax1.plot(p_acc_uncertain_list)
    ax1.set_ylabel("Probability (%)")
    ax1.set_xlabel("Certainty Threshold (%)")
    ax1.legend(["P(Accurate | Certain)", "P(Accurate | Uncertain)"])

    ax2.plot(p_certain_list)
    ax2.plot(p_uncertain_list)
    ax2.set_ylabel("Probability (%)")
    ax2.set_xlabel("Certainty Threshold (%)")
    ax2.legend(["P(Certain)", "P(Uncertain)"])

    if save_file is not None:
        split_path = save_file.split("/")
        if split_path >1 and split_path[0] not in os.listdir():
            save_folder = split_path[0]
            os.mkdir(save_folder)
        plt.savefig(save_file)

def plot_stacked_acc_certainty(p_acc_certain_list,p_acc_uncertain_list,p_certain_list,p_uncertain_list,save_file=None):
    fig, ax1 = plt.subplots(figsize=(10,15))
    ax1.plot(p_acc_certain_list)
    ax1.plot(p_acc_uncertain_list)
    ax1.set_ylabel("Probability (%)")
    ax1.set_xlabel("Certainty Threshold (%)")
    ax1.plot(p_certain_list)
    ax1.legend(["P(Accurate | Certain)", "P(Accurate | Uncertain)","P(Certain)"])

    if save_file is not None:
        split_path = save_file.split("/")
        if len(split_path) >1 and split_path[0] not in os.listdir():
            save_folder = split_path[0]
            os.mkdir(save_folder)
        plt.savefig(save_file)

def plot_histogram_acc_unc(y_res):
    acc_certainty_pd = y_res[y_res['accurate'] == True].certainty
    inacc_certainty_pd = y_res[y_res['accurate'] == False].certainty

    fig, ax1 = plt.subplots(figsize=(10,15))
    ax1.hist(acc_certainty_pd)
    ax1.hist(inacc_certainty_pd)
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Certainty (%)")
    # ax1.plot(p_certain_list)
    ax1.legend(["Accurate", "Inaccurate"])


agt_net = AgentNetwork()
monitor_agent = agt_net.get_agent("MonitorAgent_2")

architecture_string=""
for i in agt_net.get_agent("BNN_Agent_1").get_attr("architecture"):
    architecture_string += (i+"_")
bnn_architecture = "bnn_" +architecture_string

for evaluator_num in np.arange(1,6):
    memory = monitor_agent.get_attr("memory")['EvaluatorAgent_'+str(evaluator_num)]
    y_res = memory['y_res']
    #for loops
    uncertainty_thresholds = np.arange(1,101)
    p_acc_certain_list = []
    p_acc_uncertain_list = []
    p_certain_list =[]
    p_uncertain_list =[]

    for index,val in enumerate(uncertainty_thresholds):
        p_acc_certain, p_acc_uncertain = get_p_accuracy_certainty(y_res,val)
        p_certain, p_uncertain = get_p_certain(y_res,val)

        p_acc_certain_list.append(p_acc_certain)
        p_acc_uncertain_list.append(p_acc_uncertain)

        p_certain_list.append(p_certain)
        p_uncertain_list.append(p_uncertain)

    plot_stacked_acc_certainty(p_acc_certain_list,p_acc_uncertain_list,p_certain_list,p_uncertain_list,
                               save_file="plots_"+bnn_architecture+"/"+str(evaluator_num)+".png")

    #plot_subplots_acc_certainty(p_acc_certain_list,p_acc_uncertain_list,p_certain_list,p_uncertain_list)

    

