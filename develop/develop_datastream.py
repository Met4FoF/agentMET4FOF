from skmultiflow.data.base_stream import Stream
from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStream

import numpy as np
import pandas as pd
import time
#ZEMA DATA LOAD
from pandas import Series
from matplotlib import pyplot as plt
import h5py

class DataStreamMET4FOF(Stream):
    def __init__(self, x=None, y=None):
        super().__init__()
        self.sample_idx = 0
        self.current_sample_x = None
        self.current_sample_y = None

        if x is None and y is None:
            self.x = list(np.arange(10))
            self.y = list(np.arange(10))
            self.y.reverse()
        else:
            self.x = x
            self.y = y

        if type(self.x).__name__ == "list":
            self.n_samples = len(self.x)
        elif type(self.x).__name__ == "DataFrame": #dataframe or numpy
            self.x = self.x.to_numpy()
            self.n_samples = self.x.shape[0]
        elif type(self.x).__name__ == "ndarray":
            self.n_samples = self.x.shape[0]

    def prepare_for_use(self):
        self.reset()

    def next_sample(self, batch_size=1):
        self.sample_idx += batch_size

        try:
            self.current_sample_x = self.x[self.sample_idx - batch_size:self.sample_idx]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx]

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
        return {'x': self.current_sample_x, 'y': self.current_sample_y}

    def reset(self):
        self.sample_idx = 0

    def has_more_samples(self):
        return self.sample_idx < self.n_samples

class ZEMAGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        f = h5py.File("F:/PhD Research/Github/agentMet4FoF/dataset/Sensor_data_2kHz.h5", 'r')

        #prepare sensor data
        list(f.keys())
        data= f['Sensor_Data']
        data= data[:,:,:data.shape[2]-1] #drop last cycle
        data_inputs_np = np.zeros([data.shape[2],data.shape[1],data.shape[0]])
        for i in range(data.shape[0]):
            sensor_dt = data[i].transpose()
            data_inputs_np[:,:,i] = sensor_dt

        #prepare target var
        target=list(np.zeros(data_inputs_np.shape[0]))          # Making the target list which takes into account number of cycles, which-
        for i in range(data_inputs_np.shape[0]):                # goes from 0 to 100, and has number of elements same as number of cycles.
            target[i]=(i/(data_inputs_np.shape[0]-1))*100

        target_matrix = pd.DataFrame(target)        # Transforming list "target" into data frame "target matrix "
        self.stream = DataStreamMET4FOF(data_inputs_np,target_matrix)

    def agent_loop(self):
        if self.current_state == "Running":
            self.send_next_sample()

    def send_next_sample(self,num_samples=1):
        data = self.stream.next_sample(num_samples) #tuple
        self.send_output(data)

class ConvertSIAgent(AgentMET4FOF):
    def init_parameters(self):
        # Order of sensors in the picture is different from the order in imported data, which will be followed.
        self.offset=[0, 0, 0, 0, 0.00488591, 0.00488591, 0.00488591,  0.00488591, 1.36e-2, 1.5e-2, 1.09e-2]
        self.gain=[5.36e-9, 5.36e-9, 5.36e-9, 5.36e-9, 3.29e-4, 3.29e-4, 3.29e-4, 3.29e-4, 8.76e-5, 8.68e-5, 8.65e-5]
        self.b=[1, 1, 1, 1, 1, 1, 1, 1, 5.299641744, 5.299641744, 5.299641744]
        self.k=[250, 1, 10, 10, 1.25, 1, 30, 0.5, 2, 2, 2]
        self.units=['[Pa]', '[g]', '[g]', '[g]', '[kN]', '[bar]', '[mm/s]', '[A]', '[A]', '[A]', '[A]']

    def convert_SI(self,sensor_ADC):
        sensor_SI = sensor_ADC
        for i in range(sensor_ADC.shape[2]):
            sensor_SI[:,:,i]=((sensor_ADC[:,:,i]*self.gain[i])+self.offset[i])*self.b[i]*self.k[i]
        return sensor_SI

    def on_received_message(self, message):
        res = self.convert_SI(message['data']['x'])
        self.send_output({'x': res, 'y': message['data']['y']})

if __name__ == '__main__':
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType=ZEMAGeneratorAgent)
    convert_si_agent = agentNetwork.add_agent(agentType=ConvertSIAgent)
    dummy_agent = agentNetwork.add_agent(agentType=AgentMET4FOF)

    #connect agents by either way:
    agentNetwork.bind_agents(gen_agent, convert_si_agent)
    agentNetwork.bind_agents(convert_si_agent, dummy_agent)

    #gen_agent.send_next_sample(1)

    gen_agent.init_agent_loop(5)
    # # set all agents states to "Running"
    agentNetwork.set_running_state()


