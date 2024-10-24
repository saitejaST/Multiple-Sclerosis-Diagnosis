import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
import os

# Load the pre-trained model
model = load_model("VGG16.h5")
label_names = {0: 'Does not have MS', 1: 'Diagnosed with MS'}

# App title
st.title("Multiple Sclerosis Diagnosis App")

# Collect user inputs
name = st.text_input("Name:")
age = st.number_input("Age:", min_value=0, max_value=120, step=1)
sex = st.selectbox("Sex:", options=["Male", "Female", "Other"])
symptoms_duration = st.text_input("Duration of Symptoms (e.g., weeks, months):")
other_medical_history = st.text_area("Other Medical History:")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    try:
        # Open the uploaded image using PIL and convert to RGB
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert the image to a NumPy array and preprocess it
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict using the loaded model
        predictions = model.predict(image)
        predicted_label = np.argmax(predictions, axis=1)
        predicted_class = label_names[predicted_label[0]]
        st.write(f'Status: {predicted_class}')

        # Save the inputs and prediction result to a local file
        if st.button("Save Diagnosis"):
            diagnosis_info = f"""
            Name: {name}
            Age: {age}
            Sex: {sex}
            Duration of Symptoms: {symptoms_duration}
            Other Medical History: {other_medical_history}
            Predicted Status: {predicted_class}
            """
            # Define the file name and path
            output_dir = "diagnosis_results"
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_diagnosis.txt")

            # Write the information to a text file
            with open(file_path, "w") as f:
                f.write(diagnosis_info)

            st.success(f"Diagnosis saved successfully to {file_path}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


