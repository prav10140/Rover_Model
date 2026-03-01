import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Waiting for Raspberry Pi image...")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("healthy_unhealthy_model.h5")
    return model

model = load_model()

IMG_SIZE = 224

def predict_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    p = model.predict(img)[0][0]

    if p >= 0.5:
        return "Unhealthy", p * 100
    else:
        return "Healthy", (1 - p) * 100


# 🔹 Image URL input (from Raspberry Pi)
image_url = st.text_input("Image URL from Raspberry Pi")

if image_url:
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    st.image(image, caption="Received Image", use_column_width=True)

    label, confidence = predict_image(image)

    if label == "Healthy":
        st.success(f"✅ Prediction: {label}")
    else:
        st.error(f"⚠️ Prediction: {label}")

    st.write(f"Confidence: {confidence:.2f}%")
