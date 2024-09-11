import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# App header
st.write("""
# House Price Prediction App
This app predicts the **House Price**!
""")
st.write('---')

# Load the dataset
data = pd.read_csv('E:/house_prediction/BostonHousing.csv')
X = data.drop(columns=['medv'])  # Features as a DataFrame
Y = data[['medv']]  # Target variable

# Load pre-trained model from pickle file
with open('E:\house_prediction\model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar header for input parameters
st.sidebar.header('Specify Input Parameters')

# Function to get user input features
def user_input_features():
    indus = st.sidebar.slider('indus', float(X['indus'].min()), float(X['indus'].max()), float(X['indus'].mean()))
    nox = st.sidebar.slider('nox', float(X['nox'].min()), float(X['nox'].max()), float(X['nox'].mean()))
    rm = st.sidebar.slider('rm', float(X['rm'].min()), float(X['rm'].max()), float(X['rm'].mean()))
    age = st.sidebar.slider('age', float(X['age'].min()), float(X['age'].max()), float(X['age'].mean()))
    dis = st.sidebar.slider('dis', float(X['dis'].min()), float(X['dis'].max()), float(X['dis'].mean()))
    tax = st.sidebar.slider('tax', float(X['tax'].min()), float(X['tax'].max()), float(X['tax'].mean()))
    ptratio = st.sidebar.slider('ptratio', float(X['ptratio'].min()), float(X['ptratio'].max()), float(X['ptratio'].mean()))
    lstat = st.sidebar.slider('lstat', float(X['lstat'].min()), float(X['lstat'].max()), float(X['lstat'].mean()))
    
    data = {'lstat': lstat,
            'indus': indus,
            'nox': nox,
            'ptratio': ptratio,
            'rm': rm,
            'tax': tax,
            'dis': dis,
            'age': age}
    
    # Creating a DataFrame for the features in the exact order required by the model
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Ensure X contains only the features that the model was trained on
X_model_features = X[['lstat', 'indus', 'nox', 'ptratio', 'rm', 'tax', 'dis', 'age']]

# Explaining the model's predictions using SHAP values
explainer = shap.Explainer(model, X_model_features)  # Use only model features for SHAP
shap_values = explainer(df)  # Use df for prediction SHAP values

# SHAP bar plot
st.header('Feature Importance (SHAP Bar Plot)')
fig, ax = plt.subplots()
shap.summary_plot(shap_values.values, df, plot_type="bar", show=False)  # Turn off immediate display
st.pyplot(fig)

