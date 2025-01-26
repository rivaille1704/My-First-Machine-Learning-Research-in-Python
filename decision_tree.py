import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

current_directory = os.path.dirname(os.path.realpath(__file__))
csv_file_path = os.path.join(current_directory, 'dataset', 'phishing_dataset.csv')

if not os.path.exists(csv_file_path):
    print(f"File not found: {csv_file_path}")
    exit()

phishing_dataset = np.genfromtxt(csv_file_path, delimiter=',', dtype=np.int32)

samples = phishing_dataset[:, :-1]
targets = phishing_dataset[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    samples, targets, test_size=0.2, random_state=42
)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)
print("Accuracy of Decision Tree:", tree_accuracy)
print("Detailed report:")
print(classification_report(y_test, tree_predictions))
