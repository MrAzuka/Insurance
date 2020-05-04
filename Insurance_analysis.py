# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["sex"] = le.fit_transform(data["sex"])
data["smoker"] = le.fit_transform(data["smoker"])

data = pd.get_dummies(data, columns = ['region'], drop_first = True)

X = data.drop(['smoker'], axis=1)
y = data['smoker']

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X, y, train_size=0.75, random_state=101)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_val)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val,y_pred)
cm


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = rfc , X = X_val, y = y_pred, cv = 100)
accuracy.mean()