# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required Python libraries and create the datasets with study hours and marks.
2.Divide the datasets into training and testing sets.
3.Create a simple Linear Regression model and train it using the training data.
4.Use the trained model to predict marks on the testing data and display the predicted output. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: H MOHAMMED IRFAN
RegisterNumber: 212225230179 
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)
print("Dataset:\n", df.head())
df
X = df[["Hours_Studied"]] 
y = df["Marks_Scored"]   
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
![WhatsApp Image 2026-01-28 at 11 06 09 AM](https://github.com/user-attachments/assets/7fe92441-7d38-463f-a737-c7a7a033cf06)
<img width="901" height="686" alt="image" src="https://github.com/user-attachments/assets/aba8933a-8e43-4a4f-83a0-f412fe144aed" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
