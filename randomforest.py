#Import necessary modules
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
irisData = load_iris()
dataset=pd.read_csv(r"c:\\Users\\gaurav raikwar\\Download\\Iris.csv")
# Create feature and target arrays
X = irisData.data
y = irisData.target
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size = 0.4, random_state=22)

depth = np.arange(1, 9)
train_accuracy = np.empty(len(depth))
test_accuracy = np.empty(len(depth))
# Loop over depth values
for i, d in enumerate(depth):
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=d, random_state=22)
    rf.fit(X_train, y_train)
    # Compute training and test data accuracy
    train_accuracy[i] = rf.score(X_train, y_train)
    test_accuracy[i] = rf.score(X_test, y_test)

# Generate plot
plt.plot(depth, test_accuracy, label='Testing dataset Accuracy')
plt.plot(depth, train_accuracy, label='Training dataset Accuracy')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.show()