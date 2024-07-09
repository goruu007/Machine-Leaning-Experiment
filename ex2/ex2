import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Print your name and enrollment number
print("Gaurav Raikwar")
print("Enrollment no: 0901AI223D04")

# Load the Iris dataset from a CSV file
dataset = pd.read_csv(r'C:\Users\gaurav raikwar\Downloads\Iris.csv')

# Check the content of the dataset
print(dataset)

# Replace the values in the 'Species' column
dataset['Species'].replace({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}, inplace=True)

# Check the updated dataset
print(dataset)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], dataset['Species'], test_size=0.2)

# Print the lengths of the training and testing sets
print(len(X_train))
print(len(X_test))

# Create a Logistic Regression model
lr = LogisticRegression()

# Train the model using the training data
lr.fit(X_train, y_train)

# Predict the species using the test data
print(lr.predict(X_test))
