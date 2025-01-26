import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

current_directory = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_directory, 'dataset', 'smsspamcollection', 'SMSSpamCollection')

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

data = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'Message'])
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of SVM:", accuracy_svm)
print("Detailed report:")
print(classification_report(y_test, y_pred_svm))
