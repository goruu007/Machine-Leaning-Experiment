import pandas as pd
import seaborn as sns
from sklearn import linear_model
df = pd.read_csv('C:\\Users\\gaurav raikwar\\Downloads\\Churn_Modelling.csv') 
reg = linear_model.LinearRegression()
reg.fit(df[['Age']],df['EstimatedSalary'])
print(reg.predict([[200]]))
print('GAURAV RAIKWAR (0901AI223D04)')