
# Iris Flower Classification using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("Iris.csv")

# Drop the 'Id' column
df = df.drop("Id", axis=1)

# Encode the species column
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Split the data into features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:")
print(report)
