# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data from CSV file (replace the file path with your actual path)
file_path = r'C:\Users\Krutik\Desktop\Farheen-AI\CSV\water.csv'
data = pd.read_csv(file_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare the data
X_train = train_data[['Water_Consumed']]
y_train = train_data['Hydration_Level']
X_test = test_data[['Water_Consumed']]
y_test = test_data['Hydration_Level']

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

# Plot the results (replace 'Water_Consumed' and 'Hydration_Level' with your actual column names)
plt.scatter(X_test, y_test, color='black')
plt.scatter(X_test, predictions, color='red', label='Predicted Hydration Level')
plt.xlabel('Water Consumed')
plt.ylabel('Hydration Level')
plt.title('Water Consumed vs. Predicted Hydration Level')
plt.legend()
plt.show()

# Example prediction for a new water consumption value (replace 90 with your desired value)
new_water_consumed = [[1.5]]
predicted_hydration_level = model.predict(new_water_consumed)
print(f'Predicted Hydration Level for {new_water_consumed[0][0]} ounces of water consumed: {predicted_hydration_level[0]}')
