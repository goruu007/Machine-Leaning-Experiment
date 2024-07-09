import pandas as pd
df = pd.read_csv('C:\\Users\\gaurav raikwar\\Downloads\\Admission_Predict.csv')
print(df.head())
print(df.info())
print(df.describe()) # check the statical summary of data
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show
df.hist

plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual chance of Admit')
plt.ylabel('Predicted chance of Admit')
plt.show()