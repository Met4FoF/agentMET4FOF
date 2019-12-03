import os
import datetime
import pickle

class ML_Experiment:
    def __init__(self, pipelines=[], name="run", directory="ML_EXP"):
        # self.pipelines = {"agents":[],
        #                   "chains":[]}
        if type(pipelines) is not list:
            pipelines = [pipelines]

        #create new directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        #create new directory for name with running number to ensure unique runs
        running_number = 1
        name= name.lower()
        directory_lists_filter = [dir_.lower() for dir_ in os.listdir(directory) if name in dir_]
        if len(directory_lists_filter) == 0:
            name = name+"_1"
            os.makedirs(directory+"/"+name)
        else:
            directory_lists_filter = [int(dir_.split("_")[-1]) for dir_ in os.listdir(directory) if name in dir_]
            directory_lists_filter.sort()
            next_count = int(directory_lists_filter[-1])+1
            name = name+"_" +str(next_count)
            os.makedirs(directory+"/"+name)

        #record all the details
        self.name = name
        self.path = directory+"/"+self.name
        self.run_date = datetime.datetime.today()
        self.run_date_string = self.run_date.strftime("%d-%m-%y %H-%M-%S")
        self.pipeline_details = self.get_pipeline_details(pipelines)
        self.chain_results = []

    def get_pipeline_details(self, pipelines):
        pipeline_details = [pipeline.agents(ret_hyperparams=True) for pipeline in pipelines]
        return pipeline_details

    def update_chain_results(self,res):
        self.chain_results.append(res)


def save_experiment(ml_experiment):
    name = ml_experiment.name
    file = open(ml_experiment.path+"/"+name+".pkl", 'wb')
    pickle.dump(ml_experiment, file)
    file.close()

def load_experiment(ml_experiment_name,directory="ML_EXP"):
    try:
        file_path = directory+"/"+ml_experiment_name+"/"+ml_experiment_name+".pkl"
        file = open(file_path, 'rb')
        # dump information to that file
        data = pickle.load(file)
        file.close()
        return data
    except Exception as e:
        print("Error in loading experiment: "+ str(e))

    # def save_file(self):

# aggregated_results = run_experiment([data_agent_1, data_agent_2], [ML_pipelines_A,ML_pipelines_B,ML_pipeline_C], [evaluator_agent_1])
