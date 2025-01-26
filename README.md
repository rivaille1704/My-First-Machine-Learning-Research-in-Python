# My-First-Machine-Learning-Research-in-Python
I have implemented 4 Machine Learning algorithms on 2 datasets.
Some terminology:
  Precision (P): Measures how many of the predicted positive instances (spam or ham) are actually correct. A high precision means fewer false positives.

  Recall (R): Measures how many of the actual positive instances were correctly identified by the model. A high recall means fewer false negatives.

  F1-Score (F1): The harmonic mean of precision and recall. It balances both metrics and is useful when there is an imbalance between precision and recall.

  Support: The number of actual instances of each class in the dataset. It tells you how many samples belong to each class.

  Accuracy (Acc): The overall percentage of correctly predicted instances (both spam and ham) out of all predictions.

  Macro Average (Macro avg): The average of precision, recall, and F1-score across all classes, treating all classes equally regardless of their sample size.

  Weighted Average (Weighted avg): Similar to the macro average, but takes into account the number of samples in each class, giving more weight to the classes with more samples.

To be able to run on many different computers, I have edited some of the input code.
![image](https://github.com/user-attachments/assets/a316b218-f042-42bc-a233-e8a6b9493216)
This code will allow the computer to change the working directory to the current python file, from which it can determine the exact location of the dataset.
The first two implemented algorithms: SVM and Linear Regression
Dataset: The dataset we use includes ham and spam. Most email attack data is so large that tools like machine learning are needed to detect what is spam and what is not. 
I tried using SVM and Linear regression to train the model and evaluate the results.
![image](https://github.com/user-attachments/assets/dbaf973c-2eee-4b2e-a794-e503b1ba0d84)
![image](https://github.com/user-attachments/assets/1b4560d0-f3eb-4e01-8a01-b9904a16691d)


The last two implemented algorithms: Logistic Regression and Decision Tree
Dataset: The dataset we used is a CSV file with more than 30 attribute columns and 1 result. The goal is to train the model to be able to detect fraud.
![image](https://github.com/user-attachments/assets/0bbbbddf-4fe4-4c4a-9b64-32a620c4ad4b)
![image](https://github.com/user-attachments/assets/f77d646e-3195-4884-b08f-5a577504a633)

