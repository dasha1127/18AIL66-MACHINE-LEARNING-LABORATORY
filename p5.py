from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)
# Classify a new sample
new_sample = [[6.3, 2.7, 4.9, 1.8]]   # Example new sample, you can change it to your own values

# Make predictions on the new sample
predicted_class = rf_classifier.predict(new_sample)

# Get the predicted class label
predicted_class_label = iris.target_names[predicted_class[0]]

# Print the predicted class label
print("Predicted class for the new sample:", predicted_class_label)
# Predict the classes for the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)


