#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from flask import Flask, request, jsonify
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests


# In[2]:


app = Flask(__name__)


# In[3]:


df = pd.read_csv("insurance.csv") #loadDataSet


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info() #dataFrame


# In[7]:


df.columns


# In[8]:


df.describe()


# In[9]:


plt.figure(figsize=(4,4))
sns.countplot(x='smoker', data=df)
plt.title('Smoker')
plt.show()


# In[10]:


plt.figure(figsize=(4,4))
style.use('ggplot')
sns.countplot(x='sex', data=df)
plt.title('Gender')
plt.show()


# In[ ]:





# In[11]:


plt.figure(figsize=(4,4))
sns.countplot(x='region', data=df)
plt.title('Region')
plt.show()


# In[12]:


plt.figure(figsize=(5,5))
sns.barplot(x='region', y='charges', data=df)
plt.title(' charges vs Region')


# In[13]:


plt.figure(figsize=(4,4))
sns.barplot(x='sex', y='charges',hue='smoker', data=df)
plt.title('Charges for smokers')


# In[14]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='bmi', bins=20, kde=True)
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.show()


# In[15]:


correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[16]:


df['sex'] = df['sex'].apply({'male': 0, 'female': 1}.get)
df['smoker'] = df['smoker'].apply({'yes': 1, 'no': 0}.get)
df['region'] = df['region'].apply({'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4}.get)


# In[17]:


X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


linear_regression_model = LinearRegression()
svr_model = SVR()
decision_tree_model = DecisionTreeRegressor()


# In[20]:


linear_regression_model.fit(X_train, y_train)


# In[21]:


svr_model.fit(X_train, y_train)


# In[22]:


decision_tree_model.fit(X_train, y_train)


# In[23]:


linear_regression_predictions = linear_regression_model.predict(X_test)


# In[24]:


svr_predictions = svr_model.predict(X_test)


# In[25]:


decision_tree_predictions = decision_tree_model.predict(X_test)


# In[26]:


with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(linear_regression_model, f)

with open('svr_model.pkl', 'wb') as f:
    pickle.dump(svr_model, f)

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(decision_tree_model, f)


# In[27]:


data = {'age':19 , 'sex':1 , 'bmi':27.9 , 'children':0 , 'smoker':1 , 'region':1 }
index = [0]


# In[28]:


cust_df = pd.DataFrame(data, index)


# In[29]:


print(cust_df)


# In[30]:


cust_df = cust_df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]


# In[31]:


cost_pred_linear_regression = linear_regression_model.predict(cust_df)


# In[32]:


print("Linear Regression Prediction:", cost_pred_linear_regression)


# In[33]:


cost_pred_svr = svr_model.predict(cust_df)


# In[34]:


print("SVR Prediction:", cost_pred_svr)


# In[35]:


cost_pred_decision_tree = decision_tree_model.predict(cust_df)


# In[36]:


print("Decision Tree Prediction:", cost_pred_decision_tree)


# In[37]:


linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)


# In[38]:


print("Mean Squared Error Linear Regression:", linear_regression_mse)


# In[39]:


svr_mse = mean_squared_error(y_test, svr_predictions)


# In[40]:


print("Mean Squared Error SVR:", svr_mse)


# In[41]:


decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)


# In[42]:


print("Mean Squared Error Decision Tree :", decision_tree_mse)


# In[43]:


conn = sqlite3.connect('medical_insurance.db')


# In[44]:


c = conn.cursor()


# In[45]:


c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (age INTEGER, sex INTEGER, bmi REAL, children INTEGER, smoker INTEGER, region INTEGER,
             linear_regression_prediction REAL, svr_prediction REAL, decision_tree_prediction REAL, best_model TEXT)''')


# In[46]:


@app.route('/predict', methods=['POST'])
def predict():
    prediction_result = perform_prediction()


# In[57]:


input_data = {
    'age': 25,
    'sex': 1,
    'bmi': 27.9,
    'children': 2,
    'smoker': 0,
    'region': 3
}


# In[58]:


input_values = [input_data['age'], input_data['sex'], input_data['bmi'], input_data['children'], input_data['smoker'], input_data['region']]


# In[59]:


input_data = np.array([input_values])


# In[60]:


linear_regression_prediction = linear_regression_model.predict(input_data)[0]
svr_prediction = svr_model.predict(input_data)[0]
decision_tree_prediction = decision_tree_model.predict(input_data)[0]


# In[61]:


linear_regression_mse = mean_squared_error(y_test, linear_regression_predictions)
svr_mse = mean_squared_error(y_test, svr_predictions)
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)


# In[62]:


best_model = None
best_prediction = None


# In[63]:


if linear_regression_mse <= svr_mse and linear_regression_mse <= decision_tree_mse:
        best_model = 'Linear Regression'
        best_prediction = linear_regression_prediction
elif svr_mse <= linear_regression_mse and svr_mse <= decision_tree_mse:
        best_model = 'Support Vector Regression'
        best_prediction = svr_prediction
else:
        best_model = 'Decision Tree'
        best_prediction = decision_tree_prediction


# In[64]:


c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
          (input_values[0], input_values[1], input_values[2], input_values[3], input_values[4], input_values[5],
           linear_regression_prediction, svr_prediction, decision_tree_prediction))


# In[ ]:


c.close()
conn.close()


# In[56]:


return jsonify(best_prediction)


# In[56]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:




