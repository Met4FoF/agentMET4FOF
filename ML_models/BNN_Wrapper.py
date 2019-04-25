# -*- coding: utf-8 -*-

import torch
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import random
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy
from ML_models.BBBlayers import BBBLinearFactorial
from ML_models.BBBlayers import GaussianVariationalInference
import seaborn as sns
from sklearn.metrics import f1_score

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.fftpack import fft

class BNN_Full(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(BNN_Full, self).__init__()
        self.linear = BBBLinearFactorial(input_size, (int)(input_size / 2))
        self.linear_2 = BBBLinearFactorial((int)(input_size / 2), (int)(input_size / 4))
        self.linear_3 = BBBLinearFactorial((int)(input_size / 4), (int)(input_size / 8))
        self.linear_4 = BBBLinearFactorial((int)(input_size / 8), output_size)

        # self.log_softmax = torch.nn.Softmax(1)
        layers = [self.linear, self.linear_2, self.linear_3, self.linear_4]

        self.layers = torch.nn.ModuleList(layers)

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


class ML_Wrapper():
    def __init__(self):
        self.trained_model = {}

    def train_model(self, x_train_tensor, y_train_tensor):
        self.trained_model = {}
        return self.trained_model

    def predict_model_wUnc(self, x_test_tensor, num_samples):
        pred = []
        unc = []
        confidences = []
        return [pred, unc,confidences]

class BNN_Wrapper(ML_Wrapper):
    def __init__(self,model,input_size=10,output_size=10):
        super().__init__()
        self.set_modelType(model,input_size,output_size)
        self.fitted_scalers = []

    def set_modelType(self,model,input_size,output_size):
        self.modelType = model
        self.input_size = input_size
        self.output_size = output_size
    def init_model(self,input_size=-1,output_size=-1):
        if input_size==-1:
            input_size= self.input_size
        if output_size==-1:
            output_size= self.output_size
        return self.modelType(input_size,output_size)
    
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
    
    
    def fit_normalizer(self,x_train_not_normalized_yet):
        self.fitted_scalers = [RobustScaler().fit(x_train_not_normalized_yet[:,i].reshape(-1, 1)) for i in range(x_train_not_normalized_yet.shape[-1]) ]
        return self.fitted_scalers
    def run_normalizer(self,x_train_normalized):
        for i in range(len(self.fitted_scalers)):
            x_train_normalized[:,i] = self.fitted_scalers[i].transform(x_train_normalized[:,i].reshape(-1, 1)).reshape(-1)
        return x_train_normalized
    def train_model(self,x_train,y_train,learning_rate= 0.005,num_epochs = 100,plot_losses=True):
        #fit the normalizer
        self.fit_normalizer(x_train)
        
        #normalize the data
        x_train_normalized = self.run_normalizer(x_train)
        
        #convert into tensor
        x_train_tensor = Variable(torch.from_numpy(x_train_normalized).float())
        y_train_tensor = Variable(torch.from_numpy(y_train).long())
        
        #init model
        self.trained_model = self.init_model(input_size=x_train_tensor.shape[1])
        optimizer = torch.optim.Adam(self.trained_model.parameters(), lr= learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        vi = GaussianVariationalInference(criterion)
        
        #start training for n epochs
        losses=[]
        for epoch in range(num_epochs):        
            beta=1.0/x_train_tensor.shape[0]
            beta=0
            outputs, kl =self.trained_model.probforward(x_train_tensor)
            loss = vi(outputs, y_train_tensor, kl, beta)  # Loss
            if(plot_losses):
                losses.append(loss.item())
                
            optimizer.zero_grad()
            loss.backward()  # Backward Propagation
            optimizer.step()  # Optimizer update
        plt.figure(0)
        plt.plot(losses)
        plt.show()
        plt.clf()
        return self.trained_model
    def predict_model_wUnc(self,x_test, num_samples=15):
        #run normalizer
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
    
    def evaluate_model(self,y_pred_np,y_true_np,certainties_np,certainty_threshold=0.8):

        f1_score_model = f1_score(y_true_np,y_pred_np,average="micro")
        
        y_res_pd = pd.DataFrame({"pred":y_pred_np,"actual":y_true_np,"certainty":certainties_np})
        
        #plot confusion matrix
        cm = confusion_matrix(y_true_np, y_pred_np)
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
        plt.show()
        plt.clf()
        
        #filter accorting to certainty
        certain_predictions = y_res_pd[y_res_pd['certainty']>=certainty_threshold]
        uncertain_predictions = y_res_pd[y_res_pd['certainty']<certainty_threshold]
        
        incorrect_certain_preds = len(certain_predictions[certain_predictions.pred!=certain_predictions.actual].index)
        correct_certain_preds = len(certain_predictions[certain_predictions.pred==certain_predictions.actual].index)
        incorrect_uncertain_preds = len(uncertain_predictions[uncertain_predictions.pred!=uncertain_predictions.actual].index)
        correct_uncertain_preds = len(uncertain_predictions[uncertain_predictions.pred==uncertain_predictions.actual].index)
        
        
        certainty_error_matrix_raw = np.array([[correct_certain_preds,incorrect_certain_preds],[correct_uncertain_preds,incorrect_uncertain_preds]])
        certainty_error_matrix = certainty_error_matrix_raw.astype('float') / certainty_error_matrix_raw.sum(axis=1)[:, np.newaxis]
             
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
        plt.show()
        plt.clf()
        
        
        #filter according to correctness
        correct_predictions = y_res_pd[y_res_pd.pred == y_res_pd.actual]
        incorrect_predictions = y_res_pd[y_res_pd.pred != y_res_pd.actual]
        plt.figure(0)
        plt.hist(correct_predictions.certainty)
        plt.hist(incorrect_predictions.certainty)
        plt.show()
        plt.clf()
        
        
        y_res_pd=y_res_pd.assign(certain = y_res_pd.certainty >= certainty_threshold)
        y_res_pd=y_res_pd.assign(accurate = y_res_pd['pred'] == y_res_pd['actual'])
        y_res_pd=y_res_pd.assign(broken = y_res_pd.index<100)
        
               
        return f1_score_model, certainty_error_matrix[0][0], certainty_error_matrix[1][0]
