from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd


cal_housing = fetch_california_housing()
x = pd.DataFrame(cal_housing.data,columns=cal_housing.feature_names)
y = cal_housing.target
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=1, test_size=0.2)

sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)

reg = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu",random_state=1, max_iter=2000).fit(X_trainscaled, y_train)
y_pred=reg.predict(X_testscaled)
print(y_pred)
print(y_test)
