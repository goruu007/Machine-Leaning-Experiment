

import pandas as pd
df = pd.read_csv('C:\\Users\\gaurav raikwar\\Downloads\\Admission_Predict.csv')

import matplotlib.pyplot as plt
import numpy as np

# Generate some example data
y_test = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.5, 2.2, 2.8, 3.6, 4.5])

# Create a scatter plot of y_test vs. y_pred
plt.scatter(y_test, y_pred)

# Add a diagonal line representing perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')

# Add a title and axis labels
plt.xlabel('Actual chance of Admit')
plt.ylabel('Predicted chance of Admit')

# Display the plot
plt.show()
