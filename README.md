# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step-1.Start
Step-2. Import Necessary Libraries and Load Data
Step-3. Split Dataset into Training and Testing Sets
Step-4. Train the Model Using Stochastic Gradient Descent (SGD)
Step-5. Make Predictions and Evaluate Accuracy
Step-6. Generate Confusion Matrix
Step-7.End
```

## Program:
```py
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: NARENDHARAN.M
RegisterNumber:  212223230134
*/
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:

![7-3](https://github.com/user-attachments/assets/61036d27-75a8-4314-b009-38d3655e40cd)
```
Accuracy
```
![7-1](https://github.com/user-attachments/assets/99ac6704-5c90-4205-a87e-483ddf50b8b1)

```
Confusion matix
```
![7-2](https://github.com/user-attachments/assets/b7e6805c-0480-4713-9013-79088f048fe2)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
