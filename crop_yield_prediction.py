import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the directory path and file name
directory_path = '/kaggle/input/crop-yield-prediction-dataset'
file_name = 'yield.csv'
file_path = os.path.join(directory_path, file_name)

# Load the data
new_data = pd.read_csv(file_path)

# Display the first few rows
print(new_data.head())

# Select features
X_new = new_data[['Year', 'Item']]

# Convert categorical feature to numeric
X_new = pd.get_dummies(X_new, columns=['Item'])

# Clean the data
X_new = X_new.apply(pd.to_numeric, errors='coerce')
X_new.fillna(X_new.mean(), inplace=True)

# Standardize features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Initialize and train the model
model = RandomForestRegressor()
model.fit(X_new_scaled, new_data['Value'])

# Make predictions
predictions = model.predict(X_new_scaled)

# Display predictions
print(predictions)

# Evaluate the model
mse = mean_squared_error(new_data['Value'], predictions)
print(f"Mean Squared Error: {mse}")
