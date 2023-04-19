#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset for bank loan approval
n_samples = 1000
credit_score = np.random.normal(700, 50, n_samples)
income = np.random.normal(50000, 10000, n_samples)
loan_amount = np.random.normal(50000, 10000, n_samples)
approved = np.logical_and(credit_score > 650, income > 40000, loan_amount < 60000)

data = pd.DataFrame({'Credit Score': credit_score,
                     'Income': income,
                     'Loan Amount': loan_amount,
                     'Approved': approved.astype(int)})

print(data.head())

# Split data into training and testing sets
X = data.drop('Approved', axis=1)
y = data['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




