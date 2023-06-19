import numpy as np
import pandas as pa
import dataprocess

from sklearn import preprocessing
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPRegressor, MLPClassifier

#processando os dados
m = dataprocess.data_txt_to_matrix()
(x_train, y_train, x_test, y_test) = dataprocess.generate_data(m)


Label = preprocessing.LabelEncoder()
y_train = Label.fit_transform(y_train)
y_test = Label.fit_transform(y_test)


scaler = StandardScaler()
scaler.fit(x_train)
scaler.fit(x_test)

#print(len(x_test))
#print(len(x_train))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(len(x_test))
#print((x_train))


MLP = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
MLP.fit(x_train,y_train.ravel())

predictions = MLP.predict(x_test)
#print(len(y_test))
#print(len(predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
