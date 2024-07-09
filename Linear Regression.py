

import pandas as pd
df = pd.read_csv('C:\\Users\\gaurav raikwar\\Downloads\\Admission_Predict.csv')
print(df.head())     #  check the first five rows data
print(df.info())     #   check the  data type and missing values
print(df.describe()) # check the statical summary of data

#Examine the correlation between variable
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show

# check the distribution of data
df.hist()
plt.show()

# Importing the necessary libaries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

# split the data into training and testing sets
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# create a linear regression model
model = LinearRegression()

#train the model
model.fit(x_train, y_train)

# Test the model
y_pred = model.predict(x_test)


# Evaluate the models Performance
from math import sqrt
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Absolute error:', mean_absolute_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))


# Visualize the results 
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual Chance of Admit')
plt.ylabel('predict Chance of Admit')
plt.show()