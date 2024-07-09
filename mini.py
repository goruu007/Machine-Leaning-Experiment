# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Importing confusion_matrix and classification_report
import pandas as pd
import numpy as np  # Import NumPy library

# Load time series dataset
data = load_iris()

# Convert the dataset into a DataFrame
df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                  columns=data['feature_names'] + ['target'])

# Convert the time series data into static values
df_static = df.drop('target', axis=1)
target = df['target']

# Split the dataset into train and test sets
train, test, train_target, test_target = train_test_split(df_static, target, test_size=0.2, random_state=42)

# Initialize the classifier
gnb = GaussianNB()

# Train the classifier
model = gnb.fit(train, train_target)

# Make predictions on the test set
y_predict = model.predict(test)

# Print results:
print(confusion_matrix(test_target, y_predict))
print(classification_report(test_target, y_predict))
