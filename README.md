# Credit-Card-Fraud-Detection

# PROBLEM STATEMENT
The main objectives of this project are to:
Develop an accurate and efficient fraud detection system.
Need a method that is simple and fast detecting most frauds misclassifying the least.
Utilize machine learning algorithms to improve fraud detection accuracy.

# Import all Necessary Libraries of Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn  import datasets,metrics,linear_model
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import r2_score,mean_squared_error

# Import the Datasets
df=pd.read_csv('creditcard.csv')
df

# set the max columns to none
pd.set_option('display.max_columns', None)

# Check the Shape of Datasets
df.shape

df.drop('Time',axis=1,inplace=True)

df.shape

# Data Cleaning
df.isnull().sum()

# Check the Datatypes
df.dtypes

for i in df.columns:
    print(i,df[i].sort_values().unique(),'\n',sep='\n')

# Statsistical Summary of Datasets
df.describe()

# Distribution of legit transactions & fraudulent transactions
df['Class'].value_counts()
plt.figure(figsize=(6,4))
ax=sns.countplot(x='Class',data = df)
for bars in ax.containers:
    ax.bar_label(bars)
plt.xticks(rotation=90)
plt.show()

# This Dataset is highly unblanced

# 0-->Normal Transaction
# 1-->Fraudulent Transaction
#S eparating the data for analysis
Legit=df[df.Class==0]
Fraud=df[df.Class==1]
print(Legit.shape)
print(Fraud.shape)

# Statistical Measures of the Data
Legit.Amount.describe()
Fraud.Amount.describe()

# Under-Sampling
# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# Number of Fraudulent Transactions --> 492
# Compare the values for both transactions
df.groupby('Class').mean()
Legit_sample=Legit.sample(n=492)

# Concatenating two DataFrames
new_dataset=pd.concat([Legit_sample,Fraud],axis=0)
new_dataset 
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

# Splitting the data into Features & Targets
x=new_dataset.drop(columns='Class',axis=1)
y=new_dataset['Class']

# Split the data into Training data & Testing Data
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=1)
print(x.shape, X_train.shape, X_test.shape)

models={
    'Logistic Regression':LogisticRegression(random_state=5),
    'Decision Tree':DecisionTreeClassifier(criterion='entropy',random_state=16),
    'Random Forest':RandomForestClassifier(random_state=65)
}

for name, model in models.items():
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    print(name , " : {:.2f}%".format(accuracy_score(Y_pred,Y_test)*100))
    print("\n")
    print(classification_report(Y_pred,Y_test))
    print("\n")
    sns.heatmap(confusion_matrix(Y_pred,Y_test),fmt='g',annot=True)
    plt.show()
    
lg=LogisticRegression(random_state=5)
lg.fit(X_train,Y_train)

import pickle

# Store the logistic regression model using pickle
pickle.dump(lg, open('logistic_regression_model.pkl', 'wb'))

# Load the stored model
loaded_model=pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Create a test sample
pred1 = np.array([2.08271, 0.083512, -1.348547, 0.355365, 0.364725, -0.731189, 0.064821, -0.350401, 2.022140, -0.481939, 0.187867, -2.157240, 1.843239, 1.882716, 0.203100, -0.329656, 0.194048, 0.313403, -0.421817, -0.264484, 0.114721, 0.739802, 0.018036, 0.569343, 0.332178, -0.481858, -0.019221, -0.054609, 1.0])

# Reshape the input data for prediction
pred1 = pred1.reshape(1, -1)

# Make the prediction
prediction = loaded_model.predict(pred1)

# Print the predicted class
if prediction[0] == 1:
    print('Fraud')
else:
    print('Legit')



