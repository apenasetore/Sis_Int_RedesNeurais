import numpy as np
import pandas as pd
import dataprocess
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics

mlps = {}

for i in range(32):
    #processando os dados
    m = dataprocess.data_txt_to_matrix()
    (x_train, y_train, x_test, y_test) = dataprocess.generate_data(m)
    
    print(x_test[10])

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)
    scaler.fit(y_test)
    y_test = scaler.transform(y_test)
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)    
   
    

    MLP = MLPRegressor(hidden_layer_sizes=(16,32,64,32,16), max_iter= 2048, learning_rate_init= 0.0128)
    MLP.fit(x_train,y_train)

    predictions = MLP.predict(x_test)
    mrse = metrics.mean_squared_error(y_test, predictions)

    mlps[mrse] = MLP


mlps = dict(sorted(mlps.items()))

erro_medio = list(mlps)[len(mlps)//2]
print(erro_medio)
print(mlps[erro_medio])
