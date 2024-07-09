import pandas as pd
df = pd.read_csv('C:\\Users\\gaurav raikwar\\Downloads\\Admission_Predict.csv')
from math import sqrt
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Absolute error:', mean_absolute_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))