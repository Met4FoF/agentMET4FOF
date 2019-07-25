from AgentMET4FOF import AgentMET4FOF
from DataStreamMET4FOF import DataStreamMET4FOF
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
        self.fft_bfc = FFT_BFC(perc_feat=perc_feat)

    def on_received_message(self, message):
        if message['channel'] == 'train':
            res = self.fft_bfc.fit_transform(message['data']['x'])
            self.send_output({'x': res, 'y': message['data']['y']}, channel='train')
            self.send_plot(self.fft_bfc.plot_bestFreq())

        elif message['channel'] == 'test':
            res = self.fft_bfc.transform(message['data']['x'])
            self.send_output({'x': res, 'y': message['data']['y']}, channel='test')

class TrainTestSplitAgent(AgentMET4FOF):
    def init_parameters(self, train_ratio=0.8):
        self.train_ratio = train_ratio

    def on_received_message(self, message):
        x_data = message['data']['x']
        y_data = message['data']['y']
        x_train, x_test =train_test_split(x_data, train_size=self.train_ratio,random_state=15)
        y_train, y_test =train_test_split(y_data, train_size=self.train_ratio,random_state=15)

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
            self.lda_test_score = self.ml_model.score(message['data']['x'], y_true)
            
class Regression_Agent(AgentMET4FOF):
    def init_parameters(self, regression_model):      # Maybe use'BayesianRidge' as a default !! 
        if regression_model=="BayesianRidge":
            self.lin_model = linear_model.BayesianRidge()
        elif regression_model=="RandomForest":
            self.lin_model = RandomForestRegressor(n_estimators=40)
        else:
            raise Exception("Wronly defined regression model. Available models are: 'RandomForest' and 'BayesianRidge'")

#    No need to reformat the target.
#            
#    def reformat_target(self, target_vector):
#        class_target_vector=np.ceil(target_vector[0])
#        for i in class_target_vector.index:
#            if class_target_vector[i]==0:
#                class_target_vector[i]=1                   #Fixing the zero element.
#        return np.array(class_target_vector)

    def on_received_message(self, message):
        if message['channel'] == 'train':
            y_true = message['data']['y'][0]
            self.lin_model = self.lin_model.fit(message['data']['x'], y_true)
            self.log_info("Overall Train Score: " + str(self.lin_model.score(message['data']['x'], y_true)))
        elif message['channel'] == 'test':
            y_true = message['data']['y'][0]
            y_pred = self.lin_model.predict(message['data']['x'])
            self.send_output({'y_pred': y_pred, 'y_true': y_true})
            self.log_info("Overall Test Score: " + str(self.lin_model.score(message['data']['x'], y_true)))
            self.reg_test_score = self.lin_model.score(message['data']['x'], y_true) 


class EvaluatorAgent(AgentMET4FOF):
     def on_received_message(self, message):
        y_pred = message['data']['y_pred']
        y_true = message['data']['y_true']
        error_LDA1=np.abs(y_pred- y_true)                         # Change to just error
        rmse_lda= np.sqrt(mean_squared_error(y_pred, y_true))     # Also just rmse

        self.log_info("Root mean squared error of classification is:" + str(rmse_lda))  # prediction instead of classification
        self.log_info("Error_LDA1: "+str(error_LDA1))                   # Error
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
