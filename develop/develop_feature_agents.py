from skmultiflow.data.base_stream import Stream
from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStream

from develop.develop_datastream import DataStreamMET4FOF, ZEMAGeneratorAgent, ConvertSIAgent
from develop.develop_feature_extract import FFT_BFC, Pearson_FeatureSelection

import numpy as np
import pandas as pd
import time
#ZEMA DATA LOAD
from pandas import Series
from matplotlib import pyplot as plt
import h5py

from io import BytesIO
import base64

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


class FFTAgent(AgentMET4FOF):
    def init_parameters(self, sampling_period=1):
        self.sampling_period = sampling_period                                                                   # sampling period 1 s                                      # sampling points

    def on_received_message(self, message):
        x_data = message['data']['x']
        n_of_sampling_pts = x_data.shape[1]
        freq = np.fft.rfftfreq(n_of_sampling_pts, float(self.sampling_period)/n_of_sampling_pts)   # frequency axis
        amp = np.fft.rfft(x_data[0,:,0])                                                      # amplitude axis
        self.send_output({"freq": freq, "x":np.abs(amp)})

class FFT_BFCAgent(AgentMET4FOF):
    def init_parameters(self):
        self.fft_bfc = FFT_BFC()

    def on_received_message(self, message):
        if message['channel'] == 'train':
            res = self.fft_bfc.fit_transform(message['data']['x'])
            self.send_output({'x': res, 'y': message['data']['y']}, channel='train')
            self.send_plot(self.fft_bfc.plot_bestFreq())

        elif message['channel'] == 'test':
            res = self.fft_bfc.transform(message['data']['x'])
            self.send_output({'x': res, 'y': message['data']['y']}, channel='test')


from sklearn.model_selection import train_test_split

class TrainTestSplitAgent(AgentMET4FOF):
    def init_parameters(self, train_ratio = 0.8):
        self.train_ratio = train_ratio

    def on_received_message(self, message):
        x_data = message['data']['x']
        y_data = message['data']['y']
        x_train, x_test =train_test_split(x_data, train_size=self.train_ratio,random_state=10)
        y_train, y_test =train_test_split(y_data, train_size=self.train_ratio,random_state=10)

        #so that train and test will be handled sequentially
        self.send_output({'x': x_train,'y': y_train},channel='train')
        time.sleep(2)
        self.send_output({'x': x_test,'y': y_test},channel='test')


class Pearson_FeatureSelectionAgent(AgentMET4FOF):
    def init_parameters(self):
        self.pearson_fs = Pearson_FeatureSelection()

    def on_received_message(self, message):
        if message['channel'] == 'train':
            #handle train data
            selected_features, sensor_percentages = self.pearson_fs.fit_transform(message['data']['x'], message['data']['y'])
            self.send_output({'x':np.array(selected_features),'y':message['data']['y']}, channel='train')
            self.send_plot(self.pearson_fs.plot_feature_percentages(sensor_percentages,
                                                                    labels=('Microphone',
                                                                            'Vibration plain bearing','Vibration piston rod','Vibration ball bearing',
                                                                            'Axial force','Pressure','Velocity','Active current','Motor current phase 1',
                                                                            'Motor current phase 2','Motor current phase 3')))
            #handle test data
        elif message['channel'] == 'test':
            selected_features, sensor_percentages = self.pearson_fs.transform(message['data']['x'])
            self.send_output({'x':np.array(selected_features), 'y':message['data']['y']}, channel='test')
            self.log_info("HANDLING TEST DATA NOW")

class LDA_Agent(AgentMET4FOF):
    def init_parameters(self):
        self.ml_model = LinearDiscriminantAnalysis(n_components=3,priors=None, shrinkage=None, solver='eigen')

    def reformat_target(self, target_vector):
        class_target_vector=np.ceil(target_vector[0])
        for i in class_target_vector.index:
            if class_target_vector[i]==0:
                class_target_vector[i]=1                   #Fixing the zero element.
        return np.array(class_target_vector)

    def on_received_message(self, message):

        if message['channel'] == 'train':
            y_true = self.reformat_target(message['data']['y'])
            self.ml_model = self.ml_model.fit(message['data']['x'], y_true)
            self.log_info("Overall Train Score: " + str(self.ml_model.score(message['data']['x'], y_true)))
        elif message['channel'] == 'test':
            y_true = self.reformat_target(message['data']['y'])
            y_pred = self.ml_model.predict(message['data']['x'])
            self.send_output({'y_pred':y_pred, 'y_true': y_true})
            self.log_info("Overall Test Score: " + str(self.ml_model.score(message['data']['x'], y_true)))

class EvaluatorAgent(AgentMET4FOF):
     def on_received_message(self, message):
        y_pred = message['data']['y_pred']
        y_true = message['data']['y_true']
        error_LDA1=np.abs(y_pred- y_true)
        rmse_lda= np.sqrt(mean_squared_error(y_pred, y_true))

        self.log_info("Root mean squared error of classification is:" + str(rmse_lda))
        self.log_info("Error_LDA1: "+str(error_LDA1))
        self.send_output(error_LDA1)

        #send plot
        graph_comparison = self.plot_comparison(y_true,y_pred)
        self.send_plot(graph_comparison)

     def plot_comparison(self, y_true, y_pred):
         fig, ax = plt.subplots()
         ax.scatter(y_true,y_pred)
         ax.set_title("Prediction vs True Label")
         ax.set_xlabel("Y True")
         ax.set_ylabel("Y Pred")
         return fig
if __name__ == '__main__':
    #start agent network server
    agentNetwork = AgentNetwork()

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType=ZEMAGeneratorAgent)
    convert_si_agent = agentNetwork.add_agent(agentType=ConvertSIAgent)
    train_test_split_agent = agentNetwork.add_agent(agentType=TrainTestSplitAgent)
    fft_bfc_agent = agentNetwork.add_agent(agentType=FFT_BFCAgent)
    pearson_fs_agent = agentNetwork.add_agent(agentType=Pearson_FeatureSelectionAgent)
    lda_agent = agentNetwork.add_agent(agentType=LDA_Agent)
    evaluator_agent = agentNetwork.add_agent(agentType=EvaluatorAgent)

    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #connect agents
    agentNetwork.bind_agents(gen_agent, convert_si_agent)
    agentNetwork.bind_agents(convert_si_agent, train_test_split_agent)
    agentNetwork.bind_agents(train_test_split_agent, fft_bfc_agent)
    agentNetwork.bind_agents(fft_bfc_agent, pearson_fs_agent)
    agentNetwork.bind_agents(pearson_fs_agent, lda_agent)
    agentNetwork.bind_agents(lda_agent, evaluator_agent)

    #connect to monitor agents
    agentNetwork.bind_agents(fft_bfc_agent, monitor_agent)
    agentNetwork.bind_agents(pearson_fs_agent, monitor_agent)
    agentNetwork.bind_agents(lda_agent, monitor_agent)
    agentNetwork.bind_agents(evaluator_agent, monitor_agent)

    # set all agents' state to "Running"
    gen_agent.send_next_sample(5000)

    #gen_agent.init_agent_loop(5)
    #agentNetwork.set_running_state()
