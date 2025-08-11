import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import StandardScaler
 
# Load models
@st.cache_resource
def load_models():
    tabular_model = pickle.load(open("tabular_model.pkl", "rb"))
    image_model = tf.keras.models.load_model("model.h5")
    return tabular_model, image_model
 
tabular_model, image_model = load_models()
 
# Input fields for tabular data
st.title("Multimodal PCOS Diagnosis")
 
st.header("1. Enter Tabular Data")
feature_names = ['Age', 'BMI', 'Menstrual Irregularity', 'Testesterone level', 'Antral Follicle count']  # <-- Replace with your actual feature names
tabular_input = []
for name in feature_names:
    val = st.number_input(f"{name}:", value=0.0)
    tabular_input.append(val)
 
tabular_input = np.array(tabular_input).reshape(1, -1)
 
# Image upload
st.header("2. Upload Ultrasound Image")
uploaded_image = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])
 
# Prediction
if st.button("Diagnose"):
    prediction_scores = []
 
    # Tabular prediction
    try:
        tab_pred_proba = tabular_model.predict_proba(tabular_input)[0][1]
        prediction_scores.append(tab_pred_proba)
        st.success(f"Tabular Model Prediction: {tab_pred_proba:.2f}")
    except Exception as e:
        st.warning("Tabular model prediction failed.")
        prediction_scores.append(0.5)
 
    # Image prediction
    if uploaded_image:
        try:
            img = Image.open(uploaded_image).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
 
            img_pred_proba = image_model.predict(img_array)[0][0]
            prediction_scores.append(img_pred_proba)
            st.success(f"Image Model Prediction: {img_pred_proba:.2f}")
        except Exception as e:
            st.warning("Image model prediction failed.")
            prediction_scores.append(0.5)
    else:
        st.warning("No image uploaded.")
        prediction_scores.append(0.5)
 
    # Combine predictions
    final_score = np.mean(prediction_scores)
    st.subheader(f"Combined Diagnosis Score: {final_score:.2f}")
 
    if final_score > 0.5:
        st.error("Diagnosis: PCOS Positive")
    else:
        st.success("Diagnosis: PCOS Negative")