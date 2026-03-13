# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries such as pandas, sklearn, seaborn, and matplotlib.

2.Load the dataset food_items_binary.csv.

3.Select important features (Calories, Total Fat, Saturated Fat, Sugars, Dietary Fiber, Protein).

4.Define the target variable (class).

5.Split the dataset into training and testing sets.

6.Apply StandardScaler to normalize the feature values and Create an SVM classifier (SVC) model.

7.Define a parameter grid for hyperparameter tuning.

8.Use GridSearchCV with 5-fold cross validation to find the best parameters.

9.Train the model using the training dataset and Predict the output for the test dataset.

10.Calculate accuracy and generate a classification report.

11.Display the confusion matrix using a heatmap.

## Program:
```
/*
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: AKILA S
RegisterNumber:  212225220008
*/

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target='class'
X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']
}
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model=grid_search.best_estimator_
print("Name: AKILA S")
print("Register number: 212225220008")
print("Best parameters:",grid_search.best_params_)
y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name:AKILA S")
print("Register number: 212225220008")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show() 
*/
```

## Output:
<img width="780" height="701" alt="image" src="https://github.com/user-attachments/assets/4b9cd973-1573-4b47-bd63-73faa34039cc" />
<img width="604" height="287" alt="image" src="https://github.com/user-attachments/assets/9cd8e415-eb2f-4123-9f2c-2013767ddc43" />
<img width="734" height="584" alt="image" src="https://github.com/user-attachments/assets/9778bb85-41d6-4303-b2e6-ddf700e7636a" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
