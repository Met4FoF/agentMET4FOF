
# coding: utf-8

# # Data Preprocessing
# 
# We preprocess and resample the raw data .txt files we downloaded earlier into numpy. 
# 
# 
# 

# ## Resample 10Hz and 100Hz data to 1Hz

# In[4]:


import numpy as np

data_path = "Dataset/ZEMA_Hydraulic/"

filenames_input_data_10Hz = ["fs1","fs2"]       
filenames_input_data_10Hz = [file + ".txt" for file in filenames_input_data_10Hz]

filenames_input_data_100Hz = ["ps1","ps2","ps3","ps4","ps5","ps6","eps1"]       
filenames_input_data_100Hz = [file + ".txt" for file in filenames_input_data_100Hz]
  
data_input_data_10Hz = np.zeros((2205,600,len(filenames_input_data_10Hz)))
data_input_data_100Hz = np.zeros((2205,6000,len(filenames_input_data_100Hz)))

for id_,file_name in enumerate(filenames_input_data_10Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter = "\t")
    data_input_data_10Hz[:,:,id_] = input_data.copy()

for id_,file_name in enumerate(filenames_input_data_100Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter = "\t")
    data_input_data_100Hz[:,:,id_] = input_data.copy()

filenames_input_data_10Hz_resampled = ["res_"+file for file in filenames_input_data_10Hz]
filenames_input_data_100Hz_resampled = ["res_"+file for file in filenames_input_data_100Hz]

#resample 10Hz
resample = np.linspace(0,600-1, num =60,dtype="int")
data_resampled_10Hz=data_input_data_10Hz[:,resample,:]

#resample 100Hz 
resample = np.linspace(0,5999, num =60,dtype="int")
data_resampled_100Hz=data_input_data_100Hz[:,resample,:]

#save file
for id_,file_name in enumerate(filenames_input_data_10Hz_resampled):
    np.savetxt(data_path+file_name,data_resampled_10Hz[:,:,id_],delimiter='\t')
for id_,file_name in enumerate(filenames_input_data_100Hz_resampled):
    np.savetxt(data_path+file_name,data_resampled_100Hz[:,:,id_],delimiter='\t')



# ## Load all the 1Hz data
# 
# Load all data including the resampled sensors into numpy arrays

# In[11]:


#save data
datarows = 2205
seq_length = 60

#deal with inputs data
filenames_input_data_1Hz = ["ts1","ts2","ts3","ts4","vs1","se","res_fs1","res_fs2","res_ps1","res_ps2","res_ps3","res_ps4","res_ps5","res_ps6","res_eps1","ce","cp"]                           
filenames_input_data_1Hz = [file + ".txt" for file in filenames_input_data_1Hz]
filename_target_data = "profile.txt" 
        
data_input_data_1Hz = np.zeros((datarows,seq_length,len(filenames_input_data_1Hz)))

for id_,file_name in enumerate(filenames_input_data_1Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter = "\t")
    data_input_data_1Hz[:,:,id_] = input_data.copy()


# ## Load the target multi-target, multi-class output data 
# 
# We load them and preprocess into one hot vector

# In[12]:


#deal with output data now
targets_data = np.loadtxt(data_path+filename_target_data, delimiter = "\t")
        
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


import os

pickle_folder= "pickles"

if os.path.exists(pickle_folder) == False:
    os.mkdir(pickle_folder)

#Pickle them
pickle.dump(data_input_data_1Hz, open( pickle_folder+"/data_input_data_1Hz_full.p", "wb" ) )
pickle.dump(data_input_data_10Hz, open( pickle_folder+"/data_input_data_10Hz.p", "wb" ) )
pickle.dump(data_input_data_100Hz, open( pickle_folder+"/data_input_data_100Hz.p", "wb" ) )
pickle.dump(id2onehot_dictionaries, open( pickle_folder+"/id2onehot_dictionaries.p", "wb" ) )
pickle.dump(all_tensor_output, open( pickle_folder+"/zema_outputs.p", "wb" ) )

