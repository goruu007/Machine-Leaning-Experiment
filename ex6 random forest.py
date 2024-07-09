print("Gaurav Raikwar")
print("Enrollment No. 0901AI223D04")
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample data generation (replace with your actual dataset)
X = np.random.rand(100, 3)  # Example input features (100 samples, 3 features)
y = np.random.rand(100)     # Example output values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Perform prediction on new input data (replace new_data with your actual input)
new_data = np.random.rand(5, 3)  # Example new input data
predictions = rf_model.predict(new_data)
print("Predictions:", predictions)
