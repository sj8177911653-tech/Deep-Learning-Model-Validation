import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="Digit Recognition", page_icon="🔢")

model = tf.keras.models.load_model("baseline_model.keras")

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload a handwritten digit image (0–9).")

uploaded_file = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image).astype("float32") / 255.0
    image = image.reshape(1, 28, 28)

    prediction = model.predict(image)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    st.success(f"Predicted Digit: {digit}")
    st.metric("Confidence", f"{confidence:.2f}%")