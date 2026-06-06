import streamlit as st
import joblib
import numpy as np
import time

model = joblib.load("iris_model.pkl")
st.title("🌸 Iris Flower Predictor")
st.subheader("Features")

# 🌿 Sepal group
st.markdown("### 🍃 Sepal Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.0)

with col2:
    sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.0)

# 🌸 Petal group
st.markdown("### 💮 Petal Measurements")

col3, col4 = st.columns(2)

with col3:
    petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.5)

with col4:
    petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2)


# button
if st.button("Predict 🔃"):
    with st.spinner("🔍 Analyzing flower measurements..."):
      time.sleep(3)
      features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
      prediction = model.predict(features)[0]
      
      images = {
        "setosa": "Iris-setosa.jpg", 
        "versicolor": "Iris-versicolor.jpg",
        "virginica": "Iris_virginica.jpg"
    }
    
    #Result section
    st.success(f"Predicted Result")

    # Show image
    st.image(images[prediction], caption=prediction, width=300)

    st.info("This prediction is based on sepal and petal measurements.")


