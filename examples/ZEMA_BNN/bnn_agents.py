from agentMET4FOF.agents import AgentMET4FOF
from examples.ZEMA_EMC.zema_feature_extract import FFT_BFC, Pearson_FeatureSelection

import numpy as np
import time

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.fftpack import fft

import pandas as pd

import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler

from examples.ZEMA_BNN.ML_models.BBBlayers import BBBLinearFactorial
from examples.ZEMA_BNN.ML_models.BBBlayers import GaussianVariationalInference


class StatsFeaturesAgent(AgentMET4FOF):
    def on_received_message(self, message):
        message['data']['x'] = self.extract_features(message['data']['x'])

        #make it send with similar signature
        self.send_output(message)

    def extract_features(self,X_data):
        #get per line
        #convert to frequency domain
        freq_X = fft(X_data,axis=1)[:,1:(int)(X_data.shape[1]/2),:]
        def get_features(x_dt,axis=1):
            feat_skewness = skew(X_data,axis=1)
            feat_kurtosis = kurtosis(X_data,axis=1)
            feat_mean = np.mean(X_data,axis=1)
            feat_std = np.std(X_data,axis=1)
            feat_min = np.min(X_data,axis=1)
            feat_max = np.max(X_data,axis=1)
            feat_sum = np.sum(X_data,axis=1)
            feat_median = np.median(X_data,axis=1)
            return[[feat_skewness,feat_kurtosis,feat_mean,feat_std,feat_min,feat_max,feat_sum,feat_median],["skewness","kurtosis","mean","std","min","max","sum","median"]]
        ts_features = get_features(X_data)
        ts_features[1] = ["ts_"+val for val in ts_features[1]]


        fq_features = get_features(freq_X)
        fq_features[1] = ["fq_"+val for val in fq_features[1]]

        col_names =ts_features[1] +fq_features[1]
        feat_list =ts_features[0] +fq_features[0]

        def convert_df(np_arr,col_name="kurto",col_num=15):
            column_names = [col_name+"_"+str(i) for i in range(col_num)]
            df_feats_temp = pd.DataFrame.from_records(np_arr)
            df_feats_temp.columns = column_names
            return df_feats_temp
        for i,val in enumerate(col_names):
            if i ==0:
                df_feats = convert_df(np_arr=feat_list[i],col_name=col_names[i],col_num=X_data.shape[2])
            else:
                df_temp = convert_df(np_arr=feat_list[i],col_name=col_names[i],col_num=X_data.shape[2])
                df_feats=df_feats.join(df_temp)
        return df_feats

class BNN_Model(torch.nn.Module):
    def __init__(self, input_size, output_size, architecture=["d2","d4","d8"]):
        super(BNN_Model, self).__init__()
        self.architecture = architecture
        layers = []
        for layer_index,layer_string in enumerate(architecture):
            if layer_index ==0:
                first_layer = BBBLinearFactorial(input_size, self.calc_layer_size(input_size, layer_string))
                layers.append(first_layer)
            elif layer_index == (len(architecture)-1):
                new_layer = BBBLinearFactorial(self.calc_layer_size(input_size, architecture[layer_index-1]),
                                                self.calc_layer_size(input_size, layer_string))
                last_layer = BBBLinearFactorial(self.calc_layer_size(input_size, layer_string), output_size)
                layers.append(new_layer)
                layers.append(last_layer)
            else:
                new_layer = BBBLinearFactorial(self.calc_layer_size(input_size, architecture[layer_index-1]),
                                                self.calc_layer_size(input_size, layer_string))
                layers.append(new_layer)

        self.layers = torch.nn.ModuleList(layers)
    def calc_layer_size(self,input_size,layer_string):
        operator = layer_string[0]
        magnitude = float(layer_string[1:])
        if operator.lower() == "d":
            output_size = (int)(input_size/magnitude)
        elif operator.lower() == "x":
            output_size = (int)(input_size*magnitude)
        return output_size

    def probforward(self, x):
        # 'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl


class BNN_Agent(AgentMET4FOF):
    def init_parameters(self, model,input_size=10,output_size=10,learning_rate=0.005, num_epochs=300, selectY_col=-1, architecture=["d2","d4","d8"]):
        self.set_modelType(model,input_size,output_size)
        self.fitted_scalers = []
        self.learning_rate= learning_rate
        self.num_epochs = num_epochs
        self.selectY_col = selectY_col
        self.architecture = architecture
    def on_received_message(self, message):
        if message['channel'] == 'train':
            y_train = self.filter_multiY(message['data']['y'])
            self.train_model(message['data']['x'], y_train)
            self.send_plot(self.plot_training_loss())
        if message['channel'] == 'test':
            y_pred,certainties,confidences = self.predict_model_wUnc(message['data']['x'])
            y_true = self.filter_multiY(message['data']['y'])

            self.send_output({'y_true':y_true,'y_pred':y_pred, 'certainties':certainties, 'confidences':confidences},channel='test')

    def set_modelType(self,model,input_size,output_size):
        self.modelType = model
        self.input_size = input_size
        self.output_size = output_size

    def init_model(self,input_size=-1,output_size=-1):
        if input_size==-1:
            input_size= self.input_size
        if output_size==-1:
            output_size= self.output_size
        return self.modelType(input_size,output_size,self.architecture)

    def fit_normalizer(self,x_train_not_normalized_yet):
        self.fitted_scalers = [RobustScaler().fit(x_train_not_normalized_yet[:,i].reshape(-1, 1)) for i in range(x_train_not_normalized_yet.shape[-1]) ]
        return self.fitted_scalers

    def run_normalizer(self,x_train_normalized):
        for i in range(len(self.fitted_scalers)):
            x_train_normalized[:,i] = self.fitted_scalers[i].transform(x_train_normalized[:,i].reshape(-1, 1)).reshape(-1)
        return x_train_normalized

    def filter_multiY(self, y_data):
        if self.selectY_col != -1:
            return y_data[:,self.selectY_col]

    def plot_training_loss(self):
        fig = plt.figure(99)
        plt.plot(self.losses)
        return fig

    def train_model(self,x_train,y_train, plot_losses=True):
        #data type transformation
        self.log_info(type(x_train).__name__)
        if type(x_train) == pd.DataFrame:
            x_train = x_train.values

        self.log_info("Fitting normaliser")
        #fit the normalizer
        self.fit_normalizer(x_train)

        #normalize the data
        x_train_normalized = self.run_normalizer(x_train)

        #convert into tensor
        x_train_tensor = Variable(torch.from_numpy(x_train_normalized).float())
        y_train_tensor = Variable(torch.from_numpy(y_train).long())

        self.log_info("Initialise BNN model")
        #init model
        self.trained_model = self.init_model(input_size=x_train_tensor.shape[1])
        optimizer = torch.optim.Adam(self.trained_model.parameters(), lr= self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        vi = GaussianVariationalInference(criterion)

        #start training for n epochs
        losses=[]
        for epoch in range(self.num_epochs):
            beta=1.0/x_train_tensor.shape[0]
            beta=0
            outputs, kl =self.trained_model.probforward(x_train_tensor)
            loss = vi(outputs, y_train_tensor, kl, beta)  # Loss
            if plot_losses:
                losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()  # Backward Propagation
            optimizer.step()  # Optimizer update
        self.losses = losses
        self.log_info("Training finished")
        return self.trained_model, self.losses

    def predict_model_wUnc(self,x_test, num_samples=15):
        self.log_info("Predicting...")
        #run normalizer
        if type(x_test) == pd.DataFrame:
            x_test = x_test.values

        x_test_normalized = self.run_normalizer(x_test)

        #convert to tensor
        x_test_tensor = Variable(torch.from_numpy(x_test_normalized).float())

        accumulated = [self.trained_model.probforward(x_test_tensor)[0] for x in range(num_samples) ]
        results_fin = []
        confidences= []
        for x in range(num_samples):
            preds = torch.nn.functional.softmax(accumulated[x], dim=1)
            results = torch.topk(preds.cpu().data, k=1, dim=1)[1].view(-1)
            results = results.cpu().detach().numpy()
            results_fin.append(results)

            confidence = preds.cpu().detach().numpy()
            confidences.append(confidence)
        confidences = np.array(confidences)
        results_fin = np.array(results_fin)

        certainties = []
        y_pred_test = []
        for i in range(results_fin.shape[1]):
            unique, counts = np.unique(results_fin[:,i], return_counts=True)
            certainty = counts.max()/counts.sum()*100
            certainties.append(certainty)
            y_pred_test.append(unique[counts.argmax()])

        y_pred_test = np.array(y_pred_test)
        certainties = np.array(certainties)

        return y_pred_test,certainties,confidences

class EvaluatorAgent(AgentMET4FOF):

    def on_received_message(self, message):
        if message['channel'] =='test':
            evaluation_results = self.evaluate_model(message['data']['y_pred'],message['data']['y_true'],message['data']['certainties'])
            plots_res_list = ['fig_F1_confusion_matrix','fig_uncertainty_confusion_matrix','fig_uncertainty_confusion_matrix']
            numerical_res_list = ['y_res','f1_score','p_acc_certain','p_acc_uncertain']
            self.send_plot({key: evaluation_results[key] for key in plots_res_list})
            self.send_output({key: evaluation_results[key] for key in numerical_res_list})

    def plot_F1_confusion_matrix(self,y_pred,y_true):
        #plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cmap=plt.cm.Blues
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]), ylabel='True label', xlabel='Predicted label')
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        return fig

    def plot_uncertainty_confusion_matrix(self,certainty_error_matrix):
        cmap=plt.cm.Blues
        y_ticks_labels = ['T','F']
        x_ticks_labels = ['T','F']

        fig, ax = plt.subplots()
        im = ax.imshow(certainty_error_matrix, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        #ax.set(xticks=np.array(x_ticks), yticks=np.array(y_ticks), ylabel='Correctness', xlabel='Uncertainty')
        ax.set(xticks=np.arange(certainty_error_matrix.shape[1]),
        yticks=np.arange(certainty_error_matrix.shape[0]), ylabel='Certain', xlabel='Accurate')
        ax.set_xticklabels(x_ticks_labels)
        ax.set_yticklabels(y_ticks_labels)
        fmt = '.2f'
        thresh = certainty_error_matrix.max() / 2.
        for i in range(certainty_error_matrix.shape[0]):
            for j in range(certainty_error_matrix.shape[1]):
                ax.text(j, i, format(certainty_error_matrix[i, j], fmt), ha="center", va="center", color="black")
        fig.tight_layout()
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        return fig

    def plot_histogram_uncertainty(self, y_res_pd):
        #filter according to correctness
        correct_predictions = y_res_pd[y_res_pd.pred == y_res_pd.actual]
        incorrect_predictions = y_res_pd[y_res_pd.pred != y_res_pd.actual]

        fig = plt.figure(0)
        plt.clf()
        plt.hist(correct_predictions.certainty)
        plt.hist(incorrect_predictions.certainty)
        return fig

    def evaluate_model(self,y_pred_np,y_true_np,certainties_np,certainty_threshold=80):
        y_true_np = y_true_np.reshape(-1)

        self.log_info(y_true_np.shape)
        self.log_info(y_pred_np.shape)

        f1_score_model = f1_score(y_true_np,y_pred_np,average="micro")

        y_res_pd = pd.DataFrame({"pred":y_pred_np,"actual":y_true_np,"certainty":certainties_np})

        #filter accorting to certainty
        certain_predictions = y_res_pd[y_res_pd['certainty']>=certainty_threshold]
        uncertain_predictions = y_res_pd[y_res_pd['certainty']<certainty_threshold]

        incorrect_certain_preds = len(certain_predictions[certain_predictions.pred!=certain_predictions.actual].index)
        correct_certain_preds = len(certain_predictions[certain_predictions.pred==certain_predictions.actual].index)
        incorrect_uncertain_preds = len(uncertain_predictions[uncertain_predictions.pred!=uncertain_predictions.actual].index)
        correct_uncertain_preds = len(uncertain_predictions[uncertain_predictions.pred==uncertain_predictions.actual].index)


        certainty_error_matrix_raw = np.array([[correct_certain_preds,incorrect_certain_preds],[correct_uncertain_preds,incorrect_uncertain_preds]])
        certainty_error_matrix = certainty_error_matrix_raw.astype('float') / certainty_error_matrix_raw.sum(axis=1)[:, np.newaxis]

        fig_F1_confusion_matrix = self.plot_F1_confusion_matrix(y_pred_np,y_true_np)
        fig_uncertainty_confusion_matrix = self.plot_uncertainty_confusion_matrix(certainty_error_matrix)
        fig_histogram_uncertainty = self.plot_histogram_uncertainty(y_res_pd)

        y_res_pd=y_res_pd.assign(certain = y_res_pd.certainty >= certainty_threshold)
        y_res_pd=y_res_pd.assign(accurate = y_res_pd['pred'] == y_res_pd['actual'])
        y_res_pd=y_res_pd.assign(broken = y_res_pd.index<100)

        p_accurate_certain = certainty_error_matrix[0][0]
        p_accurate_uncertain = certainty_error_matrix[1][0]

        return {'y_res':y_res_pd,
                'f1_score':f1_score_model,
                'p_acc_certain':p_accurate_certain,
                'p_acc_uncertain':p_accurate_uncertain,
                'fig_F1_confusion_matrix':fig_F1_confusion_matrix,
                'fig_uncertainty_confusion_matrix':fig_uncertainty_confusion_matrix,
                'fig_histogram_uncertainty':fig_histogram_uncertainty}

