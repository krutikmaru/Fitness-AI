# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = r'C:\Users\Krutik\Desktop\Farheen-AI\CSV\calories.csv'
data = pd.read_csv(file_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare the data
X_train = train_data[['Calories_Burnt']]
y_train = train_data['Weight_Change']
X_test = test_data[['Calories_Burnt']]
y_test = test_data['Weight_Change']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared Value: {r2}')

# Plot the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)
plt.xlabel('Calories Burnt')
plt.ylabel('Weight Change')
plt.title('Calories Burnt vs. Weight Change Prediction')
plt.show()

new_calories_burnt = [[678]]  # Replace with your desired value

# Use the trained model to predict the corresponding weight change
predicted_weight_change = model.predict(new_calories_burnt)

# Print the prediction
print(f'Predicted Weight Change for {new_calories_burnt[0][0]} calories burnt: {predicted_weight_change[0]}')