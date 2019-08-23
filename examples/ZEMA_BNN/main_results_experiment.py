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

def save_plot(save_file=None):
    if save_file is not None:
        split_path = save_file.split("/")
        if len(split_path)> 0:
            folder_path_string = ""
            for path in split_path[0:-1]:
                folder_path_string=folder_path_string+"/"+path
            #remove first char which is '/'
            folder_path_string = folder_path_string[1:]
            if not os.path.exists(folder_path_string):
                os.mkdir(folder_path_string)

        plt.savefig(save_file)

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

    save_plot(save_file=save_file)

def plot_stacked_acc_certainty(p_acc_certain_list,p_acc_uncertain_list,p_certain_list,p_uncertain_list,save_file=None):
    fig, ax1 = plt.subplots(figsize=(10,15))
    ax1.plot(p_acc_certain_list)
    ax1.plot(p_acc_uncertain_list)
    ax1.set_ylabel("Probability (%)")
    ax1.set_xlabel("Certainty Threshold (%)")
    ax1.plot(p_certain_list)
    ax1.legend(["P(Accurate | Certain)", "P(Accurate | Uncertain)","P(Certain)"])

    save_plot(save_file=save_file)

def plot_histogram_acc_unc(y_res):
    acc_certainty_pd = y_res[y_res['accurate'] == True].certainty
    inacc_certainty_pd = y_res[y_res['accurate'] == False].certainty

    fig, ax1 = plt.subplots(figsize=(10,15))
    ax1.hist(acc_certainty_pd)
    ax1.hist(inacc_certainty_pd)
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Certainty (%)")
    ax1.legend(["Accurate", "Inaccurate"])

def get_architecture_string(agent_name="BNN_Agent_1",architecture_type="bnn"):
    architecture_string=""
    for i in agt_net.get_agent(agent_name).get_attr("architecture"):
        architecture_string += (i+"_")
    architecture_string = architecture_type+"_" +architecture_string
    return architecture_string

def get_crossover_point(p_acc_certain_list,p_certain_list,p_acc_uncertain_list):
    for index,val in enumerate(p_acc_certain_list):
        if index != (len(p_acc_certain_list)-1):
            if (p_acc_certain_list[index+1] > p_certain_list[index+1]) and \
                    (p_acc_certain_list[index] <= p_certain_list[index]):
               crossover_point = index
               new_dict = {'crossover_value':crossover_point,
                           'cross_p_acc_certain':p_acc_certain_list[crossover_point],
                           'cross_p_certain':p_certain_list[crossover_point],
                           'cross_p_uncertain':p_acc_uncertain_list[crossover_point],
                           }
               return pd.DataFrame(new_dict,index=[0])
    return np.nan

agt_net = AgentNetwork()
monitor_agent = agt_net.get_agent("MonitorAgent_2")

architecture_string=get_architecture_string(agent_name="BNN_Agent_1",architecture_type="bnn")

results_folder = "results"
if results_folder not in os.listdir():
    os.mkdir(results_folder)

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
    p_crossover_point = get_crossover_point(p_acc_certain_list,p_certain_list,p_acc_uncertain_list)

    plot_stacked_acc_certainty(p_acc_certain_list,p_acc_uncertain_list,p_certain_list,p_uncertain_list,
                               save_file=results_folder+"/"+"plots_"+architecture_string+"/"+str(evaluator_num)+".png")

    #plot_subplots_acc_certainty(p_acc_certain_list,p_acc_uncertain_list,p_certain_list,p_uncertain_list)

    #save summary results to csv, this to compare architectures' effects
    fixed_threshold = 80
    file_csv = results_folder+'/'+'EvaluatorAgent_'+str(evaluator_num)+'.csv'
    keys = ['f1_score']
    evaluator_dict = monitor_agent.get_attr("memory")['EvaluatorAgent_'+str(evaluator_num)]
    evaluator_dict = {key: evaluator_dict[key] for key in keys}
    evaluator_dict['architecture'] = architecture_string
    evaluator_dict['p_acc_certain'] = p_acc_certain_list[fixed_threshold-1]
    evaluator_dict['p_acc_uncertain'] = p_acc_uncertain_list[fixed_threshold-1]
    evaluator_dict['p_certain'] = p_certain_list[fixed_threshold-1]
    evaluator_dict['p_uncertain'] = p_uncertain_list[fixed_threshold-1]

    evaluator_df = pd.DataFrame.from_dict(evaluator_dict)

    if type(p_crossover_point) == pd.DataFrame:
        evaluator_df = evaluator_df.join(p_crossover_point)

    if not os.path.exists(file_csv):
        evaluator_df.to_csv(file_csv,index=False)
    else:
        evaluator_df_temp = pd.read_csv(file_csv)
        evaluator_df = evaluator_df_temp.append(evaluator_df)
        # evaluator_df.reset_index(inplace=True,drop=True)
        evaluator_df.to_csv(file_csv,index=False)

