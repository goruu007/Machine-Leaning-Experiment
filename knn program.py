

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv(r'C:\\Users\\gaurav raikwar\\Downloads\\Knn_dataset.csv')
dataset

x = df.iloc[:, :-1].values
y = df.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
from sklearn.preprocessing import StandardScaler-StandardScaler()
Scaler.fit(x_train)
x_train-Scaler.transform(x_train)x_test=Scaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
Knn-KNeighborsClassifier(n_neighbors=4)
Knn.fit (x_train,y_train)

print('Gaurav Raikwar')
print('Roll no.0901AI223D04')