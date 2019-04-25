
# coding: utf-8

# # Run Multi Agent System
# 
# We initialize the data source and create the agent network, once the agents are fully up and running, run the dashboard code in a separate terminal to visualize the agents.
# 

# In[1]:


import osbrain
from osbrain.agent import run_agent
from osbrain.agent import Agent

import pandas as pd
from datetime import datetime

import time

import pickle
import numpy as np
import random
from copy import copy

from Agent_models.agents import Sensor, Aggregator, Predictor, DecisionMaker, SensorNetwork

# TYPES OF AGENT
# 0 - SENSOR NETWORK
# 1 - SENSOR
# 2 - AGGREGATOR
# 3 - PREDICTOR
# 4 - DECISIONMAKER


# In[2]:


DemoMode= True
pickle_path = "pickles/"
data_input = pickle.load(open(pickle_path + "data_input_data_1Hz_full.p", "rb"))
data_output = pickle.load(open(pickle_path + "zema_outputs.p", "rb"))

X_data = data_input
Y_data = data_output
randomShuffling = True

if (randomShuffling == True):
    index_list = np.arange(X_data.shape[0])
    random.shuffle(index_list)
    Y_data = Y_data[index_list, :]
    X_data = X_data[index_list, :, :]


# ## Starting server

# In[3]:


if __name__ == '__main__':
	ns = osbrain.nameserver.run_nameserver(addr='127.0.0.1:14065')


	# ## Creating Agent
	# 
	# We firstly create a SensorNetwork Agent which enable wrapper functions and manages agents
	# 

	# In[4]:


	sensor_network = run_agent('sensor_network', base=SensorNetwork)


	# ## Sensor Agents
	# 
	# 1. Next, we create a Sensor Agent by `sensor_network.addsimsensor(type,unit)`, and store into a list `sensors`. 
	# 2. We set the data source of the Sensor Agent : `sensor_new.set_generatorDataSet(dataSet)` where dataSet is a 3-dimensional numpy array with: [Row x Sequence Length x Sensor]

	# In[5]:


	#add sensors
	sensors=[]
	sensorTypes = ["Temperature","Temperature","Temperature","Temperature","Vibration","EfficiencyFactor","VolumeFlow","VolumeFlow","Pressure","Pressure","Pressure","Pressure","Pressure","Pressure","MotorPower","CoolingEfficiency","CoolingPower"]
	sensorUnits = ["C","C","C","C","mm/s","%","l/min","l/min","bar","bar","bar","bar","bar","bar","W","%","kW"]

	for sensor_num in range(X_data.shape[2]):
		sensor_new = sensor_network.add_simsensor(type=sensorTypes[sensor_num], unit_v=sensorUnits[sensor_num])
		sensor_new.set_generatorDataSet(X_data[:,:,sensor_num])
		sensors.append(sensor_new)



	# 1. We can access the Sensor Agents stored in array `sensors` .
	# 2. Alternatively, the SensorNetwork Agent automatically keeps track of sensors added by it, we can access the list by calling `get_attr('sensor_list')`
	# 3. Here, we demonstrate a function of Sensor Agent which is `read_generator` which returns a random data row from the loaded dataset 

	# In[6]:


	#the sensors are loaded into array sensors
	sensor1 = sensors[0]
	print(len(sensors))

	#access sensors by either way
	sensor_network.get_attr('sensor_list')[0].read_generator()
	sensor1.read_generator()


	# ## Aggregator Agents
	# 
	# 1. We add an Aggregator Agent to the `sensor_network` by calling the function `.add_aggregator(sensorList)` where `sensorList` is an optional list of Sensor Agents which automatically binds the aggregator to the Sensor Agents. 
	# 
	# 2. Aggregator Agent can bind to Sensor Agent in runtime by calling `.bind_sensors(sensorList)`. 
	# 

	# In[7]:


	#add aggregators and bind them to sensors
	aggregator1 = sensor_network.add_aggregator(sensors)


	# ## Predictor Agents
	# 
	# 1. Similarly, we can add Predictor Agent by `.add_predictor(aggregator)` with the optional `aggregator` to be binded to.
	# 2. For each Predictor Agent, we load the prediction model by `.load_predictor_model(model)` where `model` is a trained ML_Wrapper with signature such as `.predict_model_wUnc(x_test,num_samples)` where `x_test` is the data input and `num_samples` is the number of samples for Monte Carlo sampling.
	# 3. Here, we load the previously pickled prediction model.

	# In[10]:


	#add predictor and bind to aggregator
	predictor1 = sensor_network.add_predictor(aggregator=aggregator1)
	predictor2 = sensor_network.add_predictor(aggregator=aggregator1)
	predictor3 = sensor_network.add_predictor(aggregator=aggregator1)
	predictor4 = sensor_network.add_predictor(aggregator=aggregator1)
	predictor5 = sensor_network.add_predictor(aggregator=aggregator1)

	#load predictor models
	predictor1.load_predictor_model(pickle.load(open("pickles/" + "bnn_wrapper_0.p", "rb")))
	predictor2.load_predictor_model(pickle.load(open("pickles/" + "bnn_wrapper_1.p", "rb")))
	predictor3.load_predictor_model(pickle.load(open("pickles/" + "bnn_wrapper_2.p", "rb")))
	predictor4.load_predictor_model(pickle.load(open("pickles/" + "bnn_wrapper_3.p", "rb")))
	predictor5.load_predictor_model(pickle.load(open("pickles/" + "bnn_wrapper_4.p", "rb")))


	# ## DecisionMaker Agent
	# 1. We add Decision Maker Agent calling `.add_decisionMaker()` on SensorNetwork agent
	# 2. The DM Agent is binded to every predictor by calling `.bind_predictor(predictor)` function

	# In[ ]:


	decisionMaker = sensor_network.add_decisionMaker()
	decisionMaker.bind_predictor(predictor1)
	decisionMaker.bind_predictor(predictor2)
	decisionMaker.bind_predictor(predictor3)
	decisionMaker.bind_predictor(predictor4)
	decisionMaker.bind_predictor(predictor5)


	# ## Demo
	# 1. For demo, we run an infinite loop which continuously runs the `.request_sensors_data()`
	# 2. Due to the bindings, the requested data will immediately be propagated to all binded Predictor Agents and to Decision Maker Agent
	# 3. While this is running, run the dashboard code in a separate terminal to visualize the multi-agent testbed

	# In[12]:


	#send request to aggregator agents for data from sensors

	if DemoMode:
		for i in range(99999999999):
			aggregator1.request_sensors_data()
			time.sleep(3)

