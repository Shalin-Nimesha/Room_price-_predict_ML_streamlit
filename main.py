import panda as pd
import joblib

# Dummy dataset
data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [10, 15, 20, 5, 8],
    'price': [100000, 150000, 200000, 250000, 270000]
}
df = pd.DataFrame(data)
X = df[['area', 'bedrooms', 'age']]
y = df['price']
model = LinearRegression()
model.fit(X, y)
# Save model
joblib.dump(model, 'model.pkl')

import streamlit as st

import numpy as np
# Load model
model = joblib.load("model.pkl")
st.title(" House Price Prediction App")
st.write("Enter house details to predict the price:")
# Inputs
area = st.number_input("Area (sq ft)", value=1000)
bedrooms = st.number_input("Number of Bedrooms", value=2, step=1)
age = st.number_input("Age of House (years)", value=10)
if st.button("Predict Price"):
 features = np.array([[area, bedrooms, age]])
 prediction = model.predict(features)

 st.success(f"Estimated House Price: ${prediction[0]:,.2f}")

