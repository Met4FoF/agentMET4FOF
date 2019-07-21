
import numpy as np
import pandas as pd

from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model
from skmultiflow.data import DataStream as DataStream


#ZEMA DATA LOAD
from pandas import Series
from matplotlib import pyplot as plt
import h5py
f = h5py.File("dataset/Sensor_data_2kHz.h5", 'r')

#prepare sensor data
list(f.keys())
data= f['Sensor_Data']
data= data[:,:,:data.shape[2]-1] #drop last cycle
data_inputs_np = np.zeros([data.shape[2],data.shape[1],data.shape[0]])
for i in range(data.shape[0]):
    sensor_dt = data[i].transpose()
    data_inputs_np[:,:,i] = sensor_dt

    #plot one of the cycle from each sensor
    plt.figure(0)
    sensor_cycle_dt = Series(sensor_dt[0])
    sensor_cycle_dt.plot()
    plt.show()

#prepare target var
target=list(np.zeros(data_inputs_np.shape[0]))          # Making the target list which takes into account number of cycles, which-
for i in range(data_inputs_np.shape[0]):                # goes from 0 to 100, and has number of elements same as number of cycles.
    target[i]=(i/(data_inputs_np.shape[0]-1))*100


target_matrix = pd.DataFrame(target)        # Transforming list "target" into data frame "target matrix "
plt.figure(1)
plt.plot(target_matrix.index, target_matrix.values)
plt.title("Approximation of the wear of cylinder", fontsize=16)
plt.xlabel("Cycle number", fontsize=12)
plt.ylabel("Percentage of wear of EMC", fontsize=12)
plt.grid()
plt.plot(data_inputs_np.shape[0], 100, '.', color='b')
plt.plot(0, 0, '.', color='b')

#x = np.zeros(5,4,3)
x = np.random.random_sample([5,4,3])
y = np.random.random_sample([5])
dt_stream = DataStream(data=x, y=y)


# class DataStream(AgentMET4FOF):
#     def init_parameters(self, n_wait=1.0, stream = WaveformGenerator(), pretrain_size = 100, max_samples = 100000, batch_size=100):
#
#         # parameters
#         # setup data stream
#         self.stream = stream
#         self.stream.prepare_for_use()
#         self.pretrain_size = pretrain_size
#         self.max_samples = max_samples
#         self.batch_size = batch_size
#
#         self.current_sample = 0
#         self.first_time = True
#
#     def agent_loop(self):
#         #if is running
#         if self.current_state == self.states[1]:
#             data = self.read_data()
#             if data is not None:
#                 self.send_output(data)
#         else:
#             return 0
#
#     def read_data(self):
#         if self.current_sample < self.max_samples:
#             # get sample
#             if (self.first_time):
#                 data = self.stream.next_sample(self.pretrain_size)
#                 self.current_sample += self.pretrain_size
#                 self.first_time = False
#             else:
#                 data = self.stream.next_sample(self.batch_size)
#                 self.current_sample += self.batch_size
#         else:
#             data = None
#
#         #log
#         self.log_info(data)
#         return {'x':data[0], 'y':data[1]}
#         #return data



# if __name__ == '__main__':
#     #start agent network
#     agentNetwork = AgentNetwork()
#
#     #init agents
#     data_stream_agent = agentNetwork.add_agent(agentType=DataStream)
#     monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
#
#     #connect agents
#     agentNetwork.bind_agents(data_stream_agent)
#
#     #test looping
#     data_stream_agent.init_agent_loop(0.5)
#
#     agentNetwork.set_running_state()
#     #agentNetwork.shutdown()



