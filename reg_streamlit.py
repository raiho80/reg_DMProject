import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

import streamlit as st

# streamlit containers
header = st.container()
output = st.container()

# streamlit body
with header:
    st.title("Regression Algorithm")
    st.text("")


df_le_reg = pd.read_csv("dataset_regression.csv")

#Using Boruta features
boruta_X = df_le_reg[["longitude", "latitude", "Laundry_count", "Hour", "TimeSpent_minutes", "Age_Range", "Humidity_percent", "Wind_kmph", "Temp_celsius", "Basket_colour"]]

# TotalSpent_RM is y
Y = df_le_reg['TotalSpent_RM']
colnames = boruta_X

# split by 80-20, random_state=10
X_train, X_test, Y_train, Y_test = train_test_split(boruta_X, Y, test_size = 0.2, random_state = 10)


## Decision Tree

#creating decision tree regressor model
regressorDT = DecisionTreeRegressor(splitter='best', max_depth = 3)
regressorDT.fit(X_train, Y_train)

Y_pred_dtr = np.round(regressorDT.predict(X_test),4)

df_dtr = pd.DataFrame({'Actual value' : Y_test, 'Predicted c' : Y_pred_dtr, 'Difference' : Y_test - Y_pred_dtr})  
df_dtr.sort_index()

a = "Mean squared error : ", round(sm.mean_squared_error(Y_test, Y_pred_dtr), 4)
b = "Mean absolute error : ", round(sm.mean_absolute_error(Y_test, Y_pred_dtr), 4)
c = "r2 score : ",r2_score(Y_test, Y_pred_dtr)

fn = boruta_X.columns
figDT, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,1), dpi = 500)
tree.plot_tree(regressorDT,
               feature_names = fn,  
               filled = True);


## Linear Regression

#creating linear regression model
regressorLR = LinearRegression()
regressorLR.fit(X_train, Y_train)

Y_pred_lr = np.round(regressorLR.predict(X_test),4)

df_lr = pd.DataFrame({'Actual value' : Y_test, 'Predicted value' : Y_pred_lr, 'Difference' : Y_test - Y_pred_lr})  
df_lr.sort_index()

d = "Mean squared error : ", round(sm.mean_squared_error(Y_test, Y_pred_lr), 4)
e = "Mean absolute error : ", round(sm.mean_absolute_error(Y_test, Y_pred_lr), 4)
f = "r2 score : ", r2_score(Y_test, Y_pred_lr)

#Visualizing actual value and predicted value of model
plt.figure(figsize = (10, 12))
figLR = plt.scatter(Y_test, Y_pred_lr)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual Money Spent vs Predicted Money Spent (in RM) ')


with output:
    st.header("Decision Tree")
    st.text("Q1")
    st.write(a)
    st.write(b)
    st.write(c)
    st.pyplot(fig=figDT.figure, clear_figure=None)

    st.header("Linear Regression")
    st.text("Q2")
    st.write(d)
    st.write(e)
    st.write(f)
    st.pyplot(fig=figLR.figure, clear_figure=None)
