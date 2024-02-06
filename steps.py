# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = r'C:\Users\Krutik\Desktop\Farheen-AI\CSV\steps.csv'
data = pd.read_csv(file_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare the data
X_train = train_data[['Steps']]
y_distance_train = train_data['Distance (km)']
y_walking_time_train = train_data['Walking Time (minutes)']

X_test = test_data[['Steps']]
y_distance_test = test_data['Distance (km)']
y_walking_time_test = test_data['Walking Time (minutes)']

# Initialize and train the linear regression model for distance
distance_model = LinearRegression()
distance_model.fit(X_train, y_distance_train)

# Make predictions on the test set for distance
distance_predictions = distance_model.predict(X_test)

# Evaluate the distance model
distance_mse = mean_squared_error(y_distance_test, distance_predictions)
distance_r2 = r2_score(y_distance_test, distance_predictions)

# Print evaluation metrics for distance
print(f'Distance Model Mean Squared Error: {distance_mse}')
print(f'Distance Model R-squared Value: {distance_r2}')

# Plot the results for distance
plt.scatter(X_test, y_distance_test, color='black')
plt.plot(X_test, distance_predictions, color='green', label='Distance Predictions')
plt.xlabel('Steps')
plt.ylabel('Distance (km)')
plt.title('Steps vs. Distance Prediction')
plt.legend()
plt.show()

# Initialize and train the linear regression model for walking time
walking_time_model = LinearRegression()
walking_time_model.fit(X_train, y_walking_time_train)

# Make predictions on the test set for walking time
walking_time_predictions = walking_time_model.predict(X_test)

# Evaluate the walking time model
walking_time_mse = mean_squared_error(y_walking_time_test, walking_time_predictions)
walking_time_r2 = r2_score(y_walking_time_test, walking_time_predictions)

# Print evaluation metrics for walking time
print(f'Walking Time Model Mean Squared Error: {walking_time_mse}')
print(f'Walking Time Model R-squared Value: {walking_time_r2}')

# Plot the results for walking time
plt.scatter(X_test, y_walking_time_test, color='black')
plt.plot(X_test, walking_time_predictions, color='blue', label='Walking Time Predictions')
plt.xlabel('Steps')
plt.ylabel('Walking Time (minutes)')
plt.title('Steps vs. Walking Time Prediction')
plt.legend()
plt.show()
