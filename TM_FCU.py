# Import necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the dataset with features and corresponding labels
# Features format: [weight (grams), size category (0=tiny, 1=medium, 2=giant), color code (0=grapes, 1=magenta, 2=golden yellow)]
# Labels: 0 = Grapes, 1 = Dragon Fruit, 2 = Jackfruit

x = np.array([
    [60, 1, 1],   # Dragon Fruit
    [10, 0, 0],   # Grapes
    [150, 2, 2],  # Jackfruit
    [60, 1, 1],   # Dragon Fruit
    [10, 0, 0],   # Grapes
    [150, 2, 2],  # Jackfruit
    [60, 1, 1],   # Dragon Fruit
    [10, 0, 0],   # Grapes
    [150, 2, 2],  # Jackfruit
])

y = np.array([1, 0, 2, 1, 0, 2, 1, 0, 2])  # Fruit labels matching the rows above

# Initialize and train the decision tree model using the full dataset
model = DecisionTreeClassifier()
model.fit(x, y)
print("The model has been trained successfully.")

# Predict fruit types for a new batch of fruits
test_fruits = np.array([
    [60, 1, 1],   # Should be Dragon Fruit
    [10, 0, 0],   # Should be Grapes
    [150, 2, 2]   # Should be Jackfruit
])
labels_predicted = model.predict(test_fruits)

# Print predicted fruit names based on the model output
fruit_names = {0: "Grapes", 1: "Dragon Fruit", 2: "Jackfruit"}
for i, fruits in enumerate(labels_predicted):
    print(f"Selected Fruit {i+1} is classified as: {fruit_names[fruits]}")

# Evaluate model performance using a train/test split (70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
splitmodel = DecisionTreeClassifier()
splitmodel.fit(x_train, y_train)
y_predict = splitmodel.predict(x_test)

# Calculate and display the accuracy score of the split model
accuracy = accuracy_score(y_test, y_predict)
print(f"Prediction accuracy of the model is: {accuracy * 100:.2f}%")
