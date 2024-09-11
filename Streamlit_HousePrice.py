import streamlit as st
import pandas as pd
import shap
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# App header
st.write("""
# House Price Prediction App
This app predicts the **House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
data = pd.read_csv('E:/house_prediction/BostonHousing.csv')
data.isnull().sum()
data.info()
X = data.drop(columns=['medv'])  # Features as a DataFrame
Y = data[['medv']]  # Target variable

# Sidebar header for input parameters
st.sidebar.header('Specify Input Parameters')

# Function to get user input features
def user_input_features():
    crim = st.sidebar.slider('CRIM', float(X['crim'].min()), float(X['crim'].max()), float(X['crim'].mean()))
    zn = st.sidebar.slider('ZN', float(X['zn'].min()), float(X['zn'].max()), float(X['zn'].mean()))
    indus = st.sidebar.slider('INDUS', float(X['indus'].min()), float(X['indus'].max()), float(X['indus'].mean()))
    chas = st.sidebar.slider('CHAS', float(X['chas'].min()), float(X['chas'].max()), float(X['chas'].mean()))
    nox = st.sidebar.slider('NOX', float(X['nox'].min()), float(X['nox'].max()), float(X['nox'].mean()))
    rm = st.sidebar.slider('RM', float(X['rm'].min()), float(X['rm'].max()), float(X['rm'].mean()))
    age = st.sidebar.slider('AGE', float(X['age'].min()), float(X['age'].max()), float(X['age'].mean()))
    dis = st.sidebar.slider('DIS', float(X['dis'].min()), float(X['dis'].max()), float(X['dis'].mean()))
    rad = st.sidebar.slider('RAD', float(X['rad'].min()), float(X['rad'].max()), float(X['rad'].mean()))
    tax = st.sidebar.slider('TAX', float(X['tax'].min()), float(X['tax'].max()), float(X['tax'].mean()))
    ptratio = st.sidebar.slider('PTRATIO', float(X['ptratio'].min()), float(X['ptratio'].max()), float(X['ptratio'].mean()))
    b = st.sidebar.slider('B', float(X['b'].min()), float(X['b'].max()), float(X['b'].mean()))
    lstat = st.sidebar.slider('LSTAT', float(X['lstat'].min()), float(X['lstat'].max()), float(X['lstat'].mean()))
    
    data = {'crim': crim,
            'zn': zn,
            'indus': indus,
            'chas': chas,
            'nox': nox,
            'rm': rm,
            'age': age,
            'dis': dis,
            'rad': rad,
            'tax': tax,
            'ptratio': ptratio,
            'b': b,
            'lstat': lstat}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y.values.ravel())  # Flatten Y to fit model

# Apply model to make predictions
prediction = model.predict(df)

# Display prediction
st.header('Prediction of MEDV (Median House Value)')
st.write(prediction[0])  # Display the single prediction value
st.write('---')

# Create a bar chart for the input values
fig, ax = plt.subplots()
ax.bar(df.columns, df.iloc[0], color='skyblue')
plt.xticks(rotation=90)
plt.ylabel('Value')
plt.title('Input Features')
st.pyplot(fig)
st.write('---')

# Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP summary plot
st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="dot")  # Use "dot" for summary plot
st.pyplot(fig, bbox_inches='tight')
st.write('---')

# SHAP bar plot
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar")  # Use "bar" for bar plot
st.pyplot(fig, bbox_inches='tight')
