import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



data = pd.read_csv("artifacts\Datasets\heart-disease.csv")

# independent and dependent feature split
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

# train test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# standardizing the dataset
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# initializing the network
network = Sequential()

# adding the first hidden layer
network.add(Dense(units=6,kernel_initializer="he_uniform",activation="relu",input_dim=13))

# adding the second hidden layer
network.add(Dense(units=4,kernel_initializer="he_uniform",activation="relu"))

# adding the output layer
network.add(Dense(units=1,kernel_initializer="glorot_uniform",activation="sigmoid"))

# optimization of the network
network.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"]) 

network_history = network.fit(x_train,y_train,validation_split=0.33,batch_size=10,epochs=100)

network.summary()

y_pred = network.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)

print(score)