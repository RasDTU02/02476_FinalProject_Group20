import requests
import streamlit as st
from PIL import Image

st.title("Rice Classification API")

uploaded_file = st.file_uploader(
    "Upload an image of rice", type=["jpg", "png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict_onnx/",
                files={"file": uploaded_file.getvalue()},
            )
            if response.status_code == 200:
                prediction = response.json().get("prediction", "Error")
                st.success(f"Predicted Class: {prediction}")
            else:
                st.error("Error occurred during classification")
