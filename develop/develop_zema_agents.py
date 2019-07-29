from AgentMET4FOF import AgentMET4FOF
from develop.develop_zema_feature_extract import FFT_BFC, Pearson_FeatureSelection

import numpy as np
import time

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
    def init_parameters(self, perc_feat=10):
        self.percentage_features = perc_feat
        self.fft_bfc = FFT_BFC(perc_feat=self.percentage_features)

    def on_received_message(self, message):
        if message['channel'] == 'train':
            res = self.fft_bfc.fit_transform(message['data']['x'])
            self.send_output({'x': res, 'y': message['data']['y']}, channel='train')
            self.send_plot(self.fft_bfc.plot_bestFreq())

        elif message['channel'] == 'test':
            res = self.fft_bfc.transform(message['data']['x'])
            self.send_output({'x': res, 'y': message['data']['y']}, channel='test')

class TrainTestSplitAgent(AgentMET4FOF):
    """
    This agent sets the data for train-test phases. There are two modes: Hold-out and Prequential (for incremental learning only)

    In hold-out mode, every batch of data is split into train-test with a prefixed ratio.
    In prequential mode, every batch of data is first fully tested and then trained.

    """
    def init_parameters(self, train_ratio=0.8):
        """
        train_ratio : float
            The ratio of training data in splitting the batch of data. The test_ratio then, is 1 - train_ratio.
            When train_ratio is -1, the mode is set to prequential, that is the whole batch of data is sent for testing and then training.
        """

        self.train_ratio = train_ratio

        if train_ratio > 0:
            self.pretrain_done = True
        else:
            self.pretrain_done = False

    def on_received_message(self, message):
        x_data = message['data']['x']
        y_data = message['data']['y']

        #leave one out
        if self.train_ratio > 0:
            x_train, x_test =train_test_split(x_data, train_size=self.train_ratio,random_state=15)
            y_train, y_test =train_test_split(y_data, train_size=self.train_ratio,random_state=15)

            #so that train and test will be handled sequentially
            self.send_output({'x': x_train, 'y': y_train}, channel='train')
            time.sleep(2)
            self.send_output({'x': x_test, 'y': y_test}, channel='test')

        #prequential
        else:
            if self.pretrain_done == False:
                self.pretrain_done= True
                self.send_output({'x': x_data, 'y': y_data}, channel='train')
                time.sleep(2)
            else:
                self.send_output({'x': x_data,'y': y_data}, channel='test')
                time.sleep(2)
                self.send_output({'x': x_data,'y': y_data}, channel='train')

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
    def init_parameters(self, incremental = True):
        self.ml_model = LinearDiscriminantAnalysis(n_components=3,priors=None, shrinkage=None, solver='eigen')
        self.incremental = incremental

    def reformat_target(self, target_vector):
        class_target_vector=np.ceil(target_vector[0])
        for i in class_target_vector.index:
            if class_target_vector[i]==0:
                class_target_vector[i]=1                   #Fixing the zero element.
        return np.array(class_target_vector)

    def on_received_message(self, message):
        self.log_info("MODE : "+ message['channel'])
        if message['channel'] == 'train':
            if self.incremental:
                #message['data']['y'] = message['data']['y'][0]
                message['data']['y'] = self.reformat_target(message['data']['y'])
                self.update_data_memory(message)
                y_true = self.memory[list(self.memory.keys())[0]]['y']
                x = np.array(self.memory[list(self.memory.keys())[0]]['x'])
            else:
                y_true = self.reformat_target(message['data']['y'])
                x = message['data']['x']
            self.ml_model = self.ml_model.fit(x, y_true)
            self.log_info("Overall Train Score: " + str(self.ml_model.score(x, y_true)))
        elif message['channel'] == 'test':
            y_true = self.reformat_target(message['data']['y'])
            y_pred = self.ml_model.predict(message['data']['x'])
            self.send_output({'y_pred':y_pred, 'y_true': y_true})
            self.log_info("Overall Test Score: " + str(self.ml_model.score(message['data']['x'], y_true)))
            self.lda_test_score = self.ml_model.score(message['data']['x'], y_true)


class Regression_Agent(AgentMET4FOF):
    def init_parameters(self, regression_model="BayesianRidge", incremental=True):
        self.incremental = incremental
        self.regression_model = regression_model

        if regression_model=="BayesianRidge":
            self.lin_model = linear_model.BayesianRidge()
        elif regression_model=="RandomForest":
            self.lin_model = RandomForestRegressor(n_estimators=40)
        else:
            raise Exception("Wrongly defined regression model. Available models are: 'RandomForest' and 'BayesianRidge'")

    def on_received_message(self, message):
        #self.log_info("Y MESSAGE "+ type(message['y']).__name__)

        if message['channel'] == 'train':
            if self.incremental:
                #message['data']['y'] = message['data']['y'][0]
                message['data']['y'] = message['data']['y'].values.ravel()
                self.update_data_memory(message)
                y_true = self.memory[list(self.memory.keys())[0]]['y']
                x = np.array(self.memory[list(self.memory.keys())[0]]['x'])
            else:
                y_true = message['data']['y'][0]
                x = message['data']['x']
            self.lin_model = self.lin_model.fit(x, y_true)
            self.log_info("Overall Train Score: " + str(self.lin_model.score(x, y_true)))
        elif message['channel'] == 'test':
            y_true = message['data']['y'][0]
            y_pred = self.lin_model.predict(message['data']['x'])
            for idx, value in enumerate(y_pred):
                if y_pred[idx]>100:
                    y_pred[idx]=100
                elif y_pred[idx]<0:
                    y_pred[idx]=0
            self.send_output({'y_pred': y_pred, 'y_true': np.array(y_true)})
            self.log_info("Overall Test Score: " + str(self.lin_model.score(message['data']['x'], y_true)))
            self.reg_test_score = self.lin_model.score(message['data']['x'], y_true)


class EvaluatorAgent(AgentMET4FOF):
     def on_received_message(self, message):
        self.update_data_memory(message)
        # y_pred = message['data']['y_pred']
        # y_true = message['data']['y_true']
        from_agent = message['from']
        y_pred = self.memory[from_agent]['y_pred']
        y_true = self.memory[from_agent]['y_true']
        error = np.abs(y_pred- y_true)
        rmse = np.sqrt(mean_squared_error(y_pred, y_true))

        self.log_info(message['from']+": Root mean squared error of classification is:" + str(rmse))
        self.send_output({message['from']: rmse})

        #send plot
        graph_comparison = self.plot_comparison(y_true, y_pred,
                                                from_agent=message['from'],
                                                sum_performance="RMSE: " + str(rmse))
        self.send_plot({message['from']:graph_comparison})

     def plot_comparison(self, y_true, y_pred, from_agent = None, sum_performance= ""):
         if from_agent is not None: #optional
            agent_name = from_agent
         else:
            agent_name = ""
         fig, ax = plt.subplots()
         ax.scatter(y_true,y_pred)
         fig.suptitle("Prediction vs True Label: " + agent_name)
         ax.set_title(sum_performance)
         ax.set_xlabel("Y True")
         ax.set_ylabel("Y Pred")
         return fig
