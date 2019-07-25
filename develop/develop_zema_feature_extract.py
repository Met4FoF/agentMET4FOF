import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class FFT_BFC():
    def __init__(self,perc_feat = 10):
        """
        Parameters
        ----------
        perc_feat : int
            Percentage of features to be extracted.
            
            Optimal and recommended percentage of features for ZEMA EMC dataset is 10
        """
        
        self.fitted = False;       
        self.perc_feat = perc_feat
        self.freq_of_sorted_values = []
        
    def fit(self,x_data, perc_feat = None):
        #if perc_feat is not provided explicitly, use the internal perc_feat provided during init
        #otherwise, update the internal perc_feat to the provided one
        if perc_feat is not None:
            self.perc_feat = perc_feat
        
        #get top N% frequencies AND amplitudes N=10
        n_sensors = x_data.shape[2]
        self.n_sensors = n_sensors
        n_of_samples=x_data.shape[1]

        # Initialising the list with number of sensors
        self.freq_of_sorted_values= [0]*n_sensors

        for i in range(n_sensors):
            # print("Sensor number %s" % i)
            # print("---------------------------------------------------------------------------------")
            #frequency and amplitudes
            self.freq_of_sorted_values[i],__ =self.extractBFC_train(sensor = x_data[:,:,i], n_of_samples = n_of_samples, N = self.perc_feat)
        return self
        
    def fit_transform(self,x_data, perc_feat = None):
        """
        Gets the best `perc_feat`% of features from an array with dimensions (n_of_cycles, length_cycle, n_sensors)
        
        Parameters
        ----------
        
        x_data : np.ndarray
            Sensor measurements array with dimensions (n_of_cycles, length_cycle, n_sensors)
        
        Returns
        -------
        
        sorted_values : list of DataFrames
            len(sorted_values) == n_sensors
            DataFrame.shape == (n_of_cycles, perc_feat% * length_cycle/2) 
            
        """
        
        #if perc_feat is not provided explicitly, use the internal perc_feat provided during init
        #otherwise, update the internal perc_feat to the provided one
        if perc_feat is not None:
            self.perc_feat = perc_feat
            
        n_sensors = x_data.shape[2]
        self.n_sensors = n_sensors
        n_of_samples=x_data.shape[1]

        # Initialising the list with number of sensors
        self.freq_of_sorted_values= [0 for i in range(n_sensors)]
        sorted_values_from_all_sensors= [0 for i in range(n_sensors)]

        for i in range(n_sensors):
            # print("Sensor number %s" % i)
            # print("---------------------------------------------------------------------------------")
            #frequency and amplitudes
            self.freq_of_sorted_values[i],sorted_values_from_all_sensors[i]=self.extractBFC_train(sensor = x_data[:,:,i], n_of_samples = n_of_samples, N = self.perc_feat)
        return sorted_values_from_all_sensors
    
    def transform(self,x_data):
        """
        Based on the fitted values stored in get_best_frequencies(), gets the best `perc_feat`% of features from an array with dimensions (n_of_cycles, length_cycle, n_sensors)
        
        Parameters
        ----------
        
        x_data : np.ndarray
            Sensor measurements array with dimensions (n_of_cycles, length_cycle, n_sensors)
        
        Returns
        -------
        
        sorted_values : list of DataFrames
            len(sorted_values) == n_sensors
            DataFrame.shape == (n_of_cycles, perc_feat% * length_cycle/2) 
            
        """
        
        # Storing selected features from the test data into a list "sorted_values_test"
        n_sensors = x_data.shape[2]
        n_of_samples=x_data.shape[1]

        sorted_values_test=[0]*n_sensors
        
        for i in range(n_sensors):         
            # print("Sensor number %s" % i)
            # print("---------------------------------------------------------------------------------")
            sorted_values_test[i]=self.extractBFC_test(sensor_test =x_data[:,:,i], n_of_samples = n_of_samples, N = self.perc_feat, frequencies=self.get_best_frequencies()[i])
        return sorted_values_test
      
    def get_best_frequencies(self, concise=False):
        """
        Best frequencies extracted & sorted with decreasing importance via BFC. Only available after fitting to data.
        
        Parameters
        ----------
        concise : boolean (Default = False)
            Returns a ndarray version of the best frequencies if True
        
        Returns
        -------
        
        freq_of_sorted_values : list of DataFrames
            List with length of n_sensors, and each data frame is dimension of 1 x n_features
            n_features is determined by percentage of features to be obtained during initialization of params (perc_feat). 
        
        """
        
        if concise == True:
            best_f = self.freq_of_sorted_values
            best_f_np = best_f[0].values
            if len(best_f) > 1:
                for n in range(1,len(best_f)):
                    best_f_np = np.concatenate((best_f_np,best_f[n].values))
            return best_f_np
        
        else: 
            return self.freq_of_sorted_values
    
    def extractBFC_train(self,sensor, n_of_samples, N):
        sensor = sensor.transpose((1,0))
        x_measurements=range(sensor.shape[0])                 # Number of measurements samples in time period.
        x = np.true_divide(x_measurements, n_of_samples)      # Time values, used  as real time axis.
        freq = np.fft.rfftfreq(x.size, 0.0005)                # Frequency axis, can be used for ploting in frequency domain.
        fft_amplitudes = np.fft.rfft(sensor,n_of_samples,0)   # Ndarray of amplitudes after fourier transform.
        fft_matrix = pd.DataFrame(fft_amplitudes)             # Transforming amplitudes into data frame (matrix)-
                                                              # -where one column represents amplitudes of one-
                                                              # -cycle.
        fft_matrix=fft_matrix.transpose()                     # Transposing to matrix where rows are cycles.
        n_rows, n_columns = np.shape(fft_matrix)

        print("\nNumber of cycles is: %s, and number of features is: %s" % (n_rows, n_columns))
        fft_matrix.columns = freq                    # Column labels are frequencies.

        # Calculating the average of absolute vales for each frequency (column).
        absolute_average_values_from_columns=(np.abs(fft_matrix)).mean()

        # Sorting the fft_matrix by the average of absolute vales for each frequency (column).
        fft_matrix=fft_matrix.reindex((np.abs(fft_matrix)).mean().sort_values(ascending=False).index, axis=1)

        # Taking first N percent columns from sorted fft_matrix.
        sorted_values_matrix=fft_matrix.iloc[:,:round((N/100.0)*len(freq))]

        n_rows, n_columns = np.shape(sorted_values_matrix)
        print("\nNumber of cycles is: %s, and number of selected features is: %s" % (n_rows, n_columns))
        print(np.shape(sorted_values_matrix))

        # Informations about the selected frequencies are columns in sorted data frame.
        freq_of_sorted_values=(pd.DataFrame(sorted_values_matrix.columns)).transpose()
        print("\nFirst 10 selected frequencies are:\n\n %s" % freq_of_sorted_values.values[:,:10])

        sorted_values_matrix.columns=range(round((N/100.0)*len(freq))) # Resetting the column labels.
        print("---------------------------------------------------------------------------------\n")
        # Output "sorted_values_matrix" is data frame whose rows-
        # -are cycles and columns are selected frequencies. For example,-
        # -value at position (i,j) is amplitude for frequency j in cycle i.

        return freq_of_sorted_values, sorted_values_matrix;


    def extractBFC_test(self,sensor_test, frequencies, n_of_samples, N): 
        sensor_test = sensor_test.transpose((1,0))
        x_measurements=range(sensor_test.shape[0])            # Number of measurements in time period.
        x = np.true_divide(x_measurements, n_of_samples)      # Time values, real time axis.
        freq = np.fft.rfftfreq(x.size, 0.0005)                
        fft_amplitudes = np.fft.rfft(sensor_test,n_of_samples,0)   
        fft_matrix = pd.DataFrame(fft_amplitudes)             

        fft_matrix=fft_matrix.transpose()                     # Transposing to matrix where rows are cycles.

        n_rows, n_columns = np.shape(fft_matrix)

        print("\nNumber of cycles is: %s, and number of features is: %s \n" % (n_rows, n_columns))
        fft_matrix.columns = freq                    # Column labels are frequencies.

        print("Frequencies are the same as in the traning data, of course. \nFirst 10 of them:\n\n %s" % frequencies.values[:,:10])

        sorted_values_matrix_test=fft_matrix.loc[:, frequencies.loc[0,:]]

        n_rows, n_columns = np.shape(sorted_values_matrix_test)
        print("\nNumber of cycles is: %s, and number of selected features is: %s \n\n" % (n_rows, n_columns))

        sorted_values_matrix_test.columns=range(len(sorted_values_matrix_test.columns))

        return sorted_values_matrix_test;

    def plot_feature_percentages(self, sensor_percentage, labels = None, figsize=(8,8)):
        """
        Plot pie chart which shows the percentages of features from each sensor

        """
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.set_title("Percentages of features from each sensor")
        ax1.pie(sensor_percentage, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, )
        ax1.axis('equal')

        return fig1

    def plot_bestFreq(self):
        # plot best frequencies
        best_freq = self.get_best_frequencies(concise=True)

        x=[]
        y=[]
        c=[]

        for n_sensor in range(best_freq.shape[0]):
            new_x = [n_sensor for n in range(best_freq.shape[1])]
            new_y = list(best_freq[n_sensor,:])
            new_c = list(range(best_freq.shape[1]))
            x += new_x
            y += new_y
            c += new_c

        fig = plt.figure(figsize=(8,8))
        plt.scatter(x,y,c=c, cmap="viridis")
        plt.title("Best frequencies for each sensor")
        plt.xlabel('N-th Sensor')
        plt.ylabel('Frequency')
        cbar = plt.colorbar()
        cbar.ax.invert_yaxis()
        cbar.set_label('Rank', rotation=270)
        #plt.show()
        return fig

class Pearson_FeatureSelection():
    def __init__(self, n_of_features = 500):
        self.sensor_indices = []
        self.feature_indices = []

        # Defining how much of features with biggest Pearson correllation coeff. will be selected.
        # "How many features out of %s you want to select (recommended is 500): " % n_features_for_select)
        self.n_of_features = n_of_features

    def fit(self, x_data, y_data):
        sorted_values_from_all_sensors = x_data
        n_sensors = len(sorted_values_from_all_sensors)
        n_input_features = sorted_values_from_all_sensors[0].shape[1] #100

        n_features_for_select=0
        for i in range(len(sorted_values_from_all_sensors)):
            n_features_for_select=n_features_for_select+int(len(sorted_values_from_all_sensors[i].iloc[0][:]))

        target_matrix = y_data

        print("\nDimension of target matrix is:")
        print("                                                 ", target_matrix.shape)
        print("Dimension of amplitude matrix for one sensor is:")
        print("                                                 ", sorted_values_from_all_sensors[0].iloc[:][:].shape)

        corr=list(range(n_sensors))                      # Making list for correlation coefficients.
        p_value=list(range(n_sensors))

        for j in range(n_sensors):                       # Making sublists in "corr" for each sensor.
            corr[j]=list(range(n_input_features))
            p_value[j]=list(range(n_input_features))

        # Calculating correlation coefficients for each column of each sensor with respect to target.
        for j in range(n_sensors):
            for i in range(n_input_features):
                corr[j][i],p_value[j][i]=pearsonr(np.abs(sorted_values_from_all_sensors[j].iloc[:][i]),target_matrix[0])
        #matrix_corr_coeff = np.transpose(pd.DataFrame(corr))# Transforming list of correlation coefficients to data frame.
        corr_array=np.array(corr)                                   # Transforming list of correlation coefficients to nparray

        print("Array of correlation coefficients has size:")
        print("                                                 ",corr_array.shape)

        def largest_indices(array, n):                               # Function that find indices for 500 biggest Pearson-
            """Returns the n largest indices from a numpy array."""  # -correlation coefficients.
            flat = array.flatten()
            indices = np.argpartition(flat, -n)[-n:]
            indices = indices[np.argsort(-flat[indices])]
            return np.unravel_index(indices, array.shape)

        # sensor_indices is the index of the sensor number.
        # feature_indices is the index of the feature number for each sensor number.
        sensor_indices, feature_indices = largest_indices(corr_array, self.n_of_features)

        print("Sensor indices of location of features in >sorted_values_from_all_sensors< matrix: \n")
        print(sensor_indices)
        print("\nColumn indices of location of features in >sorted_values_from_all_sensors< matrix: \n")
        print(feature_indices)
        self.sensor_indices = sensor_indices
        self.feature_indices = feature_indices
        return self
    
    def fit_transform(self, x_data, y_data):
        self.fit(x_data, y_data)
        abs_top_n_together_matrix, percentage = self.transform(x_data)
        return abs_top_n_together_matrix, percentage
    
    def transform(self, x_data):
        # Initialising a list of best features. 11 sublists containing features from each sensor, respectively.
        sorted_values_from_all_sensors = x_data
        n_sensors = len(sorted_values_from_all_sensors)
        top_n_features =[[] for n in range(n_sensors)]
        #NOTE: top_n_features =[[]]*int(n_sensors) doesn't work !!! 

        sensor_n = self.sensor_indices
        for i in range(n_sensors):
            for j in range(len(self.sensor_indices)):
                if self.sensor_indices[j]==i:
                    top_n_features[i].append(sorted_values_from_all_sensors[i].iloc[:][self.feature_indices[j]]);

        for i in range(n_sensors):
            for j in range(len(top_n_features[i])):
                top_n_features[i][j]=list(top_n_features[i][j])

        # Merging sublists into one list with all elements.
        top_n_together=[j for i in top_n_features for j in i]  

        top_n_together_matrix=np.transpose(pd.DataFrame(top_n_together))
        print(type(top_n_together_matrix), "\n")

        # Continue working with abosulte values.
        abs_top_n_together_matrix=np.abs(top_n_together_matrix)

        percentage=list(range(n_sensors))
        k=0
        for i in range(n_sensors):
            #print(top_n_features_matrix.shape)
            print("Number of features from sensor %2.0f is: %3.0f or  %4.2f %%" % (i, len(top_n_features[i]), len(top_n_features[i])/len(sensor_n)*100))
            percentage[i]=len(top_n_features[i])
            k=k+len(top_n_features[i])/len(self.sensor_indices)*100
        print("----------------------------------------------------")
        print("                                             %4.2f" % (k))
        
        return abs_top_n_together_matrix, percentage
    
    def plot_feature_percentages(self, sensor_percentage, labels = None, figsize=(8,8)):
        """
        Plot pie chart which shows the percentages of features from each sensor
        
        """
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.set_title("Percentages of features from each sensor")
        ax1.pie(sensor_percentage, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, )
        ax1.axis('equal')

        return fig1

