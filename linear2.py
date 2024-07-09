

import pandas as pd
df = pd.read_csv('C:\\Users\\gaurav raikwar\\Downloads\\Admission_Predict.csv')
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show
df.hist