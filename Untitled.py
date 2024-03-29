# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YZ9kBHeEh9_vlx3xHXSX7YsO3wE11aSY
"""

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

app = Flask(__name__)
conn = sqlite3.connect('insurance.db')
c = conn.cursor()

def load_dataset():
    df = pd.read_csv("insurance.csv")
    return df

def preprocess_data(df):
    df['sex'] = df['sex'].apply({'male': 0, 'female': 1}.get)
    df['smoker'] = df['smoker'].apply({'yes': 1, 'no': 0}.get)
    df['region'] = df['region'].apply({'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4}.get)
    return df

def train_models(X_train, y_train):
    linear_regression_model = LinearRegression()
    svr_model = SVR()
    decision_tree_model = DecisionTreeRegressor()

    linear_regression_model.fit(X_train, y_train)
    svr_model.fit(X_train, y_train)
    decision_tree_model.fit(X_train, y_train)

    return linear_regression_model, svr_model, decision_tree_model

def make_predictions(models, input_data):
    linear_regression_prediction = models[0].predict(input_data)[0]
    svr_prediction = models[1].predict(input_data)[0]
    decision_tree_prediction = models[2].predict(input_data)[0]

    return linear_regression_prediction, svr_prediction, decision_tree_prediction

def determine_best_model(mse_values):
    linear_regression_mse, svr_mse, decision_tree_mse = mse_values

    if linear_regression_mse <= svr_mse and linear_regression_mse <= decision_tree_mse:
        return 'Linear Regression'
    elif svr_mse <= linear_regression_mse and svr_mse <= decision_tree_mse:
        return 'Support Vector Regression'
    else:
        return 'Decision Tree'

def insert_predictions(input_values, predictions, best_model):
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (input_values[0], input_values[1], input_values[2], input_values[3], input_values[4], input_values[5],
               predictions[0], predictions[1], predictions[2], best_model))
    conn.commit()

def perform_prediction(input_data):
    input_values = [input_data['age'], input_data['sex'], input_data['bmi'], input_data['children'], input_data['smoker'], input_data['region']]
    input_data = np.array([input_values])

    linear_regression_prediction, svr_prediction, decision_tree_prediction = make_predictions(models, input_data)

    mse_values = (
        mean_squared_error(y_test, linear_regression_predictions),
        mean_squared_error(y_test, svr_predictions),
        mean_squared_error(y_test, decision_tree_predictions)
    )

    best_model = determine_best_model(mse_values)

    insert_predictions(input_values, (linear_regression_prediction, svr_prediction, decision_tree_prediction), best_model)

    return jsonify(linear_regression_prediction)

linear_regression_model, svr_model, decision_tree_model = None, None, None

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    prediction_result = perform_prediction(input_data)
    return prediction_result

if __name__ == '__main__':
    df = load_dataset()
    df = preprocess_data(df)

    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_regression_model, svr_model, decision_tree_model = train_models(X_train, y_train)

with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(linear_regression_model, f)

with open('svr_model.pkl', 'wb') as f:
        pickle.dump(svr_model, f)

with open('decision_tree_model.pkl', 'wb') as f:
        pickle.dump(decision_tree_model, f)

app.run(debug=True)