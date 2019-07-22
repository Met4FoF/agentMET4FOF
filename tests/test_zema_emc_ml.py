from develop.develop_zema_feature_extract import Pearson_FeatureSelection, FFT_BFC
from develop.develop_zema_datastream import ZEMA_DataStream
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
import numpy as np

import pytest

def test_zema_emc_ml():
    np.random.seed(100)

    #prepare components
    zema_datastream = ZEMA_DataStream()
    fft_bfc = FFT_BFC()
    pearson_fs = Pearson_FeatureSelection()
    ml_model = LinearDiscriminantAnalysis(n_components=3,priors=None, shrinkage=None, solver='eigen')

    def reformat_target(target_vector):
        class_target_vector=np.ceil(target_vector[0])
        for i in class_target_vector.index:
            if class_target_vector[i]== 0:
                class_target_vector[i]= 1                   #Fixing the zero element.
        return np.array(class_target_vector)

    #get data
    zema_data = zema_datastream.all_samples()

    #split
    x_data = zema_data['x']
    y_data = zema_data['y']

    x_train, x_test =train_test_split(x_data, train_size=0.8,random_state=15)
    y_train, y_test =train_test_split(y_data, train_size=0.8,random_state=15)

    #train
    x_train = fft_bfc.fit_transform(x_train)
    x_train, sensor_perc_train = pearson_fs.fit_transform(x_train, y_train)

    y_train = reformat_target(y_train)
    ml_model = ml_model.fit(x_train, y_train)
    print("Overall Train Score: " + str(ml_model.score(x_train, y_train)))

    #test
    x_test = fft_bfc.transform(x_test)
    x_test, sensor_perc_test = pearson_fs.transform(x_test)

    y_test = reformat_target(y_test)
    print("Overall Test Score: " + str(ml_model.score(x_test, y_test)))

    assert ml_model.score(x_test, y_test) == 0.8204924543288324
