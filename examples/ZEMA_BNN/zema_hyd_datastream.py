import sys
import numpy as np

from agentMET4FOF.streams import DataStreamMET4FOF

import os
import requests
import zipfile
import pickle

class ZEMA_Hyd_DataStream(DataStreamMET4FOF):
    url = ""
    path = ""

    def get_filename(self):
        return os.path.join(self.path, self.url.split('/')[-1])

    def do_download(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(self.get_filename(), "wb") as f:
            response = requests.get(self.url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write(
                        "\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()
    def extract_zip(self):
        zipDocument = zipfile.ZipFile(self.get_filename())
        zipDocument.extractall(self.path)

    def __init__(self,overwrite=False):

        self.url = "https://zenodo.org/record/1323611/files/data.zip"
        self.data_path = "dataset/ZEMA_Hydraulic/"
        self.path = os.path.join(os.path.dirname(__file__), self.data_path)

        # Check if the file is existing already, if not download the file.
        if os.path.isfile(self.get_filename()) and not overwrite:
            print("Data already exist.\n")
        else:
            print("Download data...")
            self.do_download()
            self.extract_zip()
            self.prepare_data()
            print("Download finished.\n")

        # ## Load data and label outputs
        pickle_path=self.data_path +"pickles"
        data_input = pickle.load( open( pickle_path+"/data_input_data_1Hz_full.p", "rb" ) )
        data_output = pickle.load( open( pickle_path+"/zema_outputs.p", "rb" ) )

        output_labels = [{0: "Optimal", 1: "Reduced", 2: "Nearly Fail"},
         {0: "Optimal", 1: "Small lag", 2: "Severe lag", 3: "Nearly Fail"},
         {0: "No Leakage", 1: "Weak Leakage", 2: "Severe Leakage"},
         {0: "Optimal", 1: "Slightly Reduced", 2: "Severely Reduced", 3: "Nearly Fail"},
         {0: "Stable", 1: "Unstable"}]
        output_sizes = [3,4,3,4,2]

        self.set_data_source(x=data_input, y=data_output)

    def prepare_data(self):

        filenames_input_data_10Hz = ["fs1","fs2"]
        filenames_input_data_10Hz = [file.upper() + ".txt" for file in filenames_input_data_10Hz]

        filenames_input_data_100Hz = ["ps1","ps2","ps3","ps4","ps5","ps6","eps1"]
        filenames_input_data_100Hz = [file.upper() + ".txt" for file in filenames_input_data_100Hz]

        data_input_data_10Hz = np.zeros((2205,600,len(filenames_input_data_10Hz)))
        data_input_data_100Hz = np.zeros((2205,6000,len(filenames_input_data_100Hz)))

        for id_,file_name in enumerate(filenames_input_data_10Hz):
            input_data = np.loadtxt(self.data_path + file_name, delimiter = "\t")
            data_input_data_10Hz[:,:,id_] = input_data.copy()

        for id_,file_name in enumerate(filenames_input_data_100Hz):
            input_data = np.loadtxt(self.data_path + file_name, delimiter = "\t")
            data_input_data_100Hz[:,:,id_] = input_data.copy()

        filenames_input_data_10Hz_resampled = ["res_".upper() +file for file in filenames_input_data_10Hz]
        filenames_input_data_100Hz_resampled = ["res_".upper() +file for file in filenames_input_data_100Hz]

        #resample 10Hz
        resample = np.linspace(0,600-1, num =60,dtype="int")
        data_resampled_10Hz=data_input_data_10Hz[:,resample,:]

        #resample 100Hz
        resample = np.linspace(0,5999, num =60,dtype="int")
        data_resampled_100Hz=data_input_data_100Hz[:,resample,:]

        #save file
        for id_,file_name in enumerate(filenames_input_data_10Hz_resampled):
            np.savetxt(self.data_path+file_name,data_resampled_10Hz[:,:,id_],delimiter='\t')
        for id_,file_name in enumerate(filenames_input_data_100Hz_resampled):
            np.savetxt(self.data_path+file_name,data_resampled_100Hz[:,:,id_],delimiter='\t')



        # ## Load all the 1Hz data
        #
        # Load all data including the resampled sensors into numpy arrays

        # In[11]:


        #save data
        datarows = 2205
        seq_length = 60

        #deal with inputs data
        filenames_input_data_1Hz = ["ts1","ts2","ts3","ts4","vs1","se","res_fs1","res_fs2","res_ps1","res_ps2","res_ps3","res_ps4","res_ps5","res_ps6","res_eps1","ce","cp"]
        filenames_input_data_1Hz = [file.upper() + ".txt" for file in filenames_input_data_1Hz]
        filename_target_data = "profile.txt"

        data_input_data_1Hz = np.zeros((datarows,seq_length,len(filenames_input_data_1Hz)))

        for id_,file_name in enumerate(filenames_input_data_1Hz):
            input_data = np.loadtxt(self.data_path + file_name, delimiter = "\t")
            data_input_data_1Hz[:,:,id_] = input_data.copy()


        # ## Load the target multi-target, multi-class output data
        #
        # We load them and preprocess into one hot vector

        # In[12]:


        #deal with output data now
        targets_data = np.loadtxt(self.data_path+filename_target_data, delimiter = "\t")

        #conversion of outputs to one hot
        def makeOneHotVectorMap(length):
            map_toOneHot ={}
            for i in range(length):
                oneHot = np.zeros(length)
                oneHot[i] = 1
                map_toOneHot[i] = oneHot
            return map_toOneHot

        id2x_dictionaries = []
        x2id_dictionaries = []
        id2onehot_dictionaries = []

        for label in range(targets_data.shape[1]):
            label_column = list(set(targets_data[:,label]))
            label_column.sort(reverse=True)
            id2x_dictionary = {}
            x2id_dictionary = {}
            id2onehot_dictionary = makeOneHotVectorMap(len(label_column))
            for i in range(len(label_column)):
                id2x_dictionary[i] = label_column[i]
                x2id_dictionary[label_column[i]] = i
            id2x_dictionaries+=[id2x_dictionary]
            x2id_dictionaries+=[x2id_dictionary]
            id2onehot_dictionaries+=[id2onehot_dictionary]

        #convert a row into one-hot coded multi-class multi-label
        onehot_tensor_output = []
        id_output =[]
        for row in range(targets_data.shape[0]):
            row_output_data= targets_data[row]
            onehots_row =[]
            id_row =[]
            for label in range(row_output_data.shape[0]):
                id_ = x2id_dictionaries[label][row_output_data[label]]
                onehot= id2onehot_dictionaries[label][id_]
                onehots_row =np.append(onehots_row,onehot)
                id_row = np.append(id_row,id_)
            id_output+=[id_row]
            onehot_tensor_output += [onehots_row]
        onehot_tensor_output = np.array(onehot_tensor_output)
        id_tensor_output = np.array(id_output)

        tensor_output = id_tensor_output
        all_tensor_output = id_tensor_output


        # ## Pickle data

        # In[14]:
        pickle_folder= self.data_path+"pickles"

        if os.path.exists(pickle_folder) == False:
            os.mkdir(pickle_folder)

        #Pickle them
        pickle.dump(data_input_data_1Hz, open( pickle_folder+"/data_input_data_1Hz_full.p", "wb" ) )
        pickle.dump(data_input_data_10Hz, open( pickle_folder+"/data_input_data_10Hz.p", "wb" ) )
        pickle.dump(data_input_data_100Hz, open( pickle_folder+"/data_input_data_100Hz.p", "wb" ) )
        pickle.dump(id2onehot_dictionaries, open( pickle_folder+"/id2onehot_dictionaries.p", "wb" ) )
        pickle.dump(all_tensor_output, open( pickle_folder+"/zema_outputs.p", "wb" ) )




"""
Examples
--------
from agentMET4FOF.agents import AgentNetwork, AgentMET4FOF, DataStreamAgent
if __name__ == '__main__':
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)
    dummy_agent = agentNetwork.add_agent(agentType=AgentMET4FOF)

    datastream_agent.init_parameters(stream=ZEMA_Hyd_DataStream(),
                                     pretrain_size=1000, batch_size=250,
                                     loop_wait=5, randomize=False)

    #connect agents by either way:
    agentNetwork.bind_agents(datastream_agent, dummy_agent)

    # # set all agents states to "Running"
    agentNetwork.set_running_state()
"""



