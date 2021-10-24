# Data Wraggling
import numpy as np
import pandas as pd
from numpy import *
# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
# Statistics
import scipy.stats as stats
# Scikit learn
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Keras 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def load_data(data_url, normalise):
    column_names = ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "rings"]

    data = pd.read_csv(data_url, sep=',', names=column_names)
    #data.head()
    
    df_org = data.copy()
    
    # Convert categorical feature to numerical one
    encoder = LabelEncoder()
    encoder.fit(data['sex'])
    data['sex'] = encoder.transform(data['sex'])
    
    # Convert rings to age
    data['age'] = data['rings'] + 1.5
    data = data.drop('rings', axis=1)
    #data.head()
    
    if normalise == True:
        # Normalise the input features and the response variable between 0 and 1
        dataset = normalize(data)        
    else:
        dataset = data
    #dataset.head()

    # Save normalised data as .csv file
    dataset.to_csv("data/abalone.csv", encoding='utf-8', index=False)
    
    data_X = dataset.values[:, :-1]
    data_y = dataset.values[:, -1] 

    return dataset, data_X, data_y, df_org

def countplot(df):
    sns.countplot(x='sex', data=df)
    plt.show()

def corr_plot(df):
    # Develop a correlation map using a heatmap
    corr_matrix = df.corr().round(2)
    plt.figure(figsize=(10,10))
    ax = sns.heatmap(corr_matrix, vmin=-1, center=0, annot=True, cmap='BrBG')
    plt.savefig('figure/corr_mat.png')
    plt.show()
    plt.clf()

def boxenplot(df):
    # Categorical features
    temp = pd.concat([df['rings'], df['sex']], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxenplot(x='sex', y="rings", data=df)
    fig.axis(ymin=0, ymax=30)
    plt.savefig('figure/boxen.png')
    plt.show()
    plt.clf()

def pairgrid(df):
    g = sns.PairGrid(df, diag_sharey=False, corner=True)
    g.map_lower(sns.scatterplot)
    g.map_diag(sns.kdeplot)
    plt.savefig('figure/pairgrid.png')
    plt.show()
    plt.clf()

def hist_plot(df):
    # Create histograms of all features
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,10))
    df.plot(kind='hist', subplots=True, ax=axes, alpha=0.5, bins=30)
    plt.savefig('figure/hist.png')
    plt.show()
    plt.clf()    
    
def split_data(data_X, data_y, split_ratio, run_num):
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=split_ratio, random_state=run_num)
    return X_train, X_test, y_train, y_test

def keras_nn(X_train, X_test, y_train, y_test, hidden, model_type, learn_rate):
    '''Neural network.
    :param hidden: Number of neurons in the hidden layer.
    :param model_type: Optimizer options: 0 -single hidden layer with SGD, 1 -single hidden layer with Adam, 2 -two hidden layers.
    :param learn_rate: Tuning parameter in an optimization algorithm.
    :returns:
        mse_train, mse_test, rmse, r2, residuals, history.
    '''    
    if model_type == 0:   # single hidden layer with SGD
        # Create the Sequential model
        model = Sequential([
            Dense(hidden, input_dim=X_train.shape[1], activation="relu", kernel_initializer='normal'),
            Dense(1, activation='normal')
        ])
        # Specify the loss function and the optimizer after a model is created
        sgd = SGD(learning_rate=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="mse", optimizer=sgd, metrics=['mse'])
            
    elif model_type == 1:   # single hidden layer with Adam
        # Create the Sequential model
        model = Sequential([
            Dense(hidden, input_dim=X_train.shape[1], activation="relu", kernel_initializer='normal'),
            Dense(1, activation='normal')
        ])
        # Specify the loss function and the optimizer after a model is created
        adam = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss="mse", optimizer=adam, metrics=['mse'])
        
    elif model_type == 2:   # two hidden layers with SGD
        # Create the Sequential model
        model = Sequential([
            Dense(hidden, input_dim=X_train.shape[1], activation="relu", kernel_initializer='normal'),
            Dense(hidden, activation="relu", kernel_initializer='normal'),
            Dense(1, activation='normal')
        ])
        # Specify the loss function and the optimizer after a model is created
        sgd = SGD(learning_rate=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="mse", optimizer=sgd, metrics=['mse'])
            
    else:
        print('no model')
        
    # Fit model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3000, batch_size=10, verbose=0)
    #history = model.fit(X_train, y_train, validation_split=0.25, epochs=3000, batch_size=10, verbose=0)

    # Evaluate the model
    # https://keras.io/api/models/model_training_apis/
    loss_train, mse_train = model.evaluate(X_train, y_train, verbose=0)
    loss_test, mse_test = model.evaluate(X_test, y_test, verbose=0)
    
    # Predit
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    residuals = y_pred - y_test
    
    return mse_train, mse_test, rmse, r2, residuals, history

def result_present(mse_train, mse_test, rmse, r2, residuals, history, task, run_num, fname):
    
    print('Experiment {} with {} -- '.format(run_num, fname) + 'MSE_train: %.3f, MSE_test: %.3f' % (mse_train, mse_test))

    print('With {} -- '.format(fname) + 'RMSE: %.3f, R2 score: %.3f' % (rmse, r2))
    
    # plot metrics
    plt.title('MSE Loss at {} run for {}'.format(run_num, fname))
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('figure/{}/MSE_{}_run_{}.png'.format(task, run_num, fname))
    plt.show()
    plt.clf()
    
    # plot residuals    
    plt.title('Residuals at {} run for {}'.format(run_num, fname))
    plt.plot(residuals, linewidth=1)
    plt.savefig('figure/{}/res_{}_run_{}.png'.format(task, run_num, fname))
    plt.show()
    plt.clf()
    

def main():

    # Task 1: Analyse and visualise the given data sets by reporting the distribution of class, distribution of features and any relevant visualisation
    # Normalise the data
    normalise = True 
    # Get all dataset and subsets
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
    dataset, data_X, data_y, df_org = load_data(data_url, normalise)
    
    # Countplot w.r.t. the categorical feature 'sex'
    countplot(df_org)
    # Boxenplot to display the distribution of data based on min, Q1, median, Q3, and max values
    boxenplot(df_org)
    # Correlation heatmap to display the dependence between features and response variables
    corr_plot(dataset)
    # Histograms to display the distribution of continuous numerical features
    hist_plot(dataset)
    # Subplot grid for plotting pairwise relationships in a dataset
    pairgrid(dataset)

    max_expruns = 10
    split_ratio = 0.4

    
    # Task 2: Investigate the effect of the number of hidden neurons (e.g. 5, 10, 15, 20) for a single hidden layer
    task = 'task2'

    hidden_lst = [5, 10, 15, 20]
    num_hidden = len(hidden_lst)
    col_names = ['5-neuron', '10-neuron', '15-neuron', '20-neuron']
    rmse_mx = np.zeros((max_expruns, num_hidden))
    r2_mx = np.zeros((max_expruns, num_hidden))
    learn_rate = 0.07
    model_type = 1   # using optimizer Adam to compare effect of number of hidden neurons
        
    for run_num in range(max_expruns):
        print("Task 2: Experiment {} running...\n".format(run_num))
        
        X_train, X_test, y_train, y_test = split_data(data_X, data_y, split_ratio, run_num)
        
        for hidden in hidden_lst:
            hidden_idx = hidden_lst.index(hidden)
            
            mse_train, mse_test, rmse, r2, residuals, history = keras_nn(X_train, X_test, y_train, y_test, hidden, model_type, learn_rate)
            
            fname = hidden + '-neuron' 
            result_present(mse_train, mse_test, rmse, r2, residuals, history, task, run_num, fname)
            
            rmse_mx[run_num, hidden_idx] = rmse
            r2_mx[run_num, hidden_idx] = r2

    rmse_df = pd.DataFrame(rmse_mx, columns=col_names)
    rmse_df.to_csv("data/task2_rmse.csv", encoding='utf-8', index=False)
    print("Task2_RMSE: \n", rmse_df)

    r2_df = pd.DataFrame(r2_mx, columns=col_names)
    r2_df.to_csv("data/task2_r2.csv", encoding='utf-8', index=False)
    print("\n Task2_R2 score: \n", r2_df)

    rmse_mean = np.zeros(num_hidden)
    rmse_std = np.zeros(num_hidden)

    r2_mean = np.zeros(num_hidden)
    r2_std = np.zeros(num_hidden)

    rmse_mean = rmse_mx.mean(axis=1)
    rmse_std = rmse_mx.std(axis=1)

    r2_mean = r2_mx.mean(axis=1)
    r2_std = r2_mx.std(axis=1)

    np.savetxt('data/task2_results.txt', (rmse_mean, rmse_std, r2_mean, r2_std), fmt='%1.5f')

    print("\n Task2_ Mean RMSE: ", rmse_mean)
    print("\n Task2_ std of RMSE: ", rmse_std)
    print("\n Task2_ Mean R-squared value: ", r2_mean)
    print("\n Task2_ std of R-squared value: ", r2_std)
    
    
    # Task 3: Investigate the effect of learning rate (in case of SGD) for the selected data set (using the optimal number of hidden neurons).
    task = 'task3'

    lr_lst = list(np.arange(0.01, 0.1, 0.02))
    num_lr = len(lr_lst)
    col_names = ['lr=0.01', 'lr=0.03', 'lr=0.05', 'lr=0.07', 'lr=0.09']
    rmse_mx = np.zeros((max_expruns, num_lr))
    r2_mx = np.zeros((max_expruns, num_lr))
    model_type = 0   # using optimizer SGD to compare effect of learning rate
    hidden = 10      # based on result of Task 2, 10 hidden neurons outperformed
        
    for run_num in range(max_expruns):
        print("Task 3: Experiment {} running...\n".format(run_num))
        
        X_train, X_test, y_train, y_test = split_data(data_X, data_y, run_num)
        
        for learn_rate in lr_lst:
            lr_idx = lr_lst.index(learn_rate)
            
            mse_train, mse_test, rmse, r2, residuals, history = keras_nn(X_train, X_test, y_train, y_test, hidden, model_type, learn_rate)
            
            fname = 'learn_rate_' + learn_rate.round(2)
            result_present(mse_train, mse_test, rmse, r2, residuals, history, task, run_num, fname)
            
            rmse_mx[run_num, lr_idx] = rmse
            r2_mx[run_num, lr_idx] = r2       

    rmse_df = pd.DataFrame(rmse_mx, columns=col_names)
    rmse_df.to_csv("data/task3_rmse.csv", encoding='utf-8', index=False)
    print("Task3_RMSE: \n", rmse_df)
    #display(rmse_df)

    r2_df = pd.DataFrame(r2_mx, columns=col_names)
    r2_df.to_csv("data/task3_r2.csv", encoding='utf-8', index=False)
    print("\n Task3_R2 score: \n", r2_df)
    #display(r2_df)

    rmse_mean = np.zeros(num_hidden)
    rmse_std = np.zeros(num_hidden)

    r2_mean = np.zeros(num_hidden)
    r2_std = np.zeros(num_hidden)

    rmse_mean = rmse_mx.mean(axis=1)
    rmse_std = rmse_mx.std(axis=1)

    r2_mean = r2_mx.mean(axis=1)
    r2_std = r2_mx.std(axis=1)

    np.savetxt('data/task3_results.txt', (rmse_mean, rmse_std, r2_mean, r2_std), fmt='%1.5f')

    print("\n Task3_ Mean RMSE: ", rmse_mean)
    print("\n Task3_ std of RMSE: ", rmse_std)
    print("\n Task3_ Mean R-squared value: ", r2_mean)
    print("\n Task3_ std of R-squared value: ", r2_std)
    

    # Task 4: Investigate the effect on a different number of hidden layers (1, 2) with the optimal number of hidden neurons (from Part 4).
    task = 'task4'

    rmse_ar = np.zeros(max_expruns)
    r2_ar = np.zeros(max_expruns)
    model_type = 2      # two hidden layers with SGD
    learn_rate = 0.07   # based on result of Task 3, learning rate ? outperformed
    hidden = 10         # based on result of Task 2, 10 hidden neurons outperformed
        
    for run_num in range(max_expruns):
        print("Task 4: Experiment {} running...\n".format(run_num))
        
        X_train, X_test, y_train, y_test = split_data(data_X, data_y, run_num)
        
        mse_train, mse_test, rmse, r2, residuals, history = keras_nn(X_train, X_test, y_train, y_test, hidden, model_type, learn_rate)
            
        fname = '2-hidden-layer'
        result_present(mse_train, mse_test, rmse, r2, residuals, history, task, run_num, fname)
            
        rmse_ar[run_num] = rmse
        r2_ar[run_num] = r2       

    print("Task4_RMSE: \n", rmse_ar)
    print("\n Task4_R2 score: \n", r2_ar)

    rmse_mean = np.mean(rmse_ar)
    rmse_std = np.std(rmse_ar)

    r2_mean = np.mean(r2_ar)
    r2_std = np.std(r2_ar)

    np.savetxt('data/task4_results.txt', (rmse_mean, rmse_std, r2_mean, r2_std), fmt='%1.5f')

    print("\n Task4_ Mean RMSE: ", rmse_mean)
    print("\n Task4_ std of RMSE: ", rmse_std)
    print("\n Task4_ Mean R-squared value: ", r2_mean)
    print("\n Task4_ std of R-squared value: ", r2_std)
    
    
    # Task 5: Investigate the effect of Adam and SGD on training and test performance.
    task = 'task5'
    
    rmse_ar = np.zeros(max_expruns)
    r2_ar = np.zeros(max_expruns)
    model_type = 1      # single hidden layer with Adam to compare to SGD done in task 3 
    learn_rate = 0.07   # based on result of Task 3, learning rate ? outperformed    
    hidden = 10         # based on result of Task 2, 10 hidden neurons outperformed
        
    for run_num in range(max_expruns):
        print("Task 5: Experiment {} running...\n".format(run_num))
        
        X_train, X_test, y_train, y_test = split_data(data_X, data_y, run_num)
        
        mse_train, mse_test, rmse, r2, residuals, history = keras_nn(X_train, X_test, y_train, y_test, hidden, model_type, learn_rate)
            
        fname = 'adam_vs_sgd'
        result_present(mse_train, mse_test, rmse, r2, residuals, history, task, run_num, fname)
            
        rmse_ar[run_num] = rmse
        r2_ar[run_num] = r2       

    print("Task5_RMSE: \n", rmse_ar)    
    print("\n Task5_R2 score: \n", r2_ar)

    rmse_mean = np.mean(rmse_ar)
    rmse_std = np.std(rmse_ar)

    r2_mean = np.mean(r2_ar)
    r2_std = np.std(r2_ar)

    np.savetxt('data/task5_results.txt', (rmse_mean, rmse_std, r2_mean, r2_std), fmt='%1.5f')

    print("\n Task5_ Mean RMSE: ", rmse_mean)
    print("\n Task5_ std of RMSE: ", rmse_std)
    print("\n Task5_ Mean R-squared value: ", r2_mean)
    print("\n Task5_ std of R-squared value: ", r2_std)
    
    
    
if __name__ == '__main__':
    main()