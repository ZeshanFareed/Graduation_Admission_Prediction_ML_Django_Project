from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

def user(request):
    return render(request, 'userinput.html')

def viewdata(request):
    # Load the dataset
    df = pd.read_csv("C:/Users/PMLS/Documents/ML/ML Algorithms/Admission_Predict.csv")
    
    df = df.drop('Serial No.' , axis = 1)
    
    # Split the data into input and output
    X = df.drop('Chance of Admit ', axis=1)
    y = df['Chance of Admit ']

    # Polynomial features transformation
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # Scaling the data
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_poly)
    
    # Perform cross-validation
    lr = LinearRegression()
    cv_scores = cross_val_score(lr, X_sc, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores = -cv_scores  # Convert negative MSE to positive MSE
  

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Polynomial features transformation for training and testing sets
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scaling the data for training and testing sets
    X_train_sc = sc.fit_transform(X_train_poly)
    X_test_sc = sc.transform(X_test_poly)

    # Train the Linear Regression model
    lr.fit(X_train_sc, y_train)

    # Retrieve input data from the GET request
    new_data = [
        int(request.GET['GRE Score']),
        int(request.GET['TOEFL Score']),
        int(request.GET['University Rating']),
        float(request.GET['SOP']),
        float(request.GET['LOR']),
        float(request.GET['CGPA']),
        int(request.GET['Research'])
    ]

    # Convert new_data to a DataFrame and reshape to 2D array
    new_data_df = pd.DataFrame([new_data])

    # Prepare the data for prediction
    new_data_poly = poly.transform(new_data_df)
    new_data_scaled = sc.transform(new_data_poly)

    # Make prediction
    y_pred = lr.predict(new_data_scaled)

    data = {
        'message': 'Your Chance of Admission = ',
        'prediction': y_pred[0],

    }

    return render(request, 'viewdata.html', data)
