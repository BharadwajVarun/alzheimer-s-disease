import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from keras.models import load_model
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import requests
from fpdf import FPDF
import base64
import io

# Load model with caching
@st.cache_resource
def load_alzheimer_model():
    model = load_model('best_finetuned_model.h5')
    return model

model = load_alzheimer_model()

st.set_page_config(page_title="Alzheimer's Detection App", layout="wide")
st.title("Alzheimer's Detection and Support System")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Upload MRI", "Symptom Questionnaire", "Hospital Locator", "Explain Alzheimer's Affected Regions"])

pdf_data = ""

# Function to highlight affected brain regions based on stage
def show_brain_regions(image, stage):
    brain_regions = {
        "hippocampus": (50, 70, 100, 120),  # Example coordinates for hippocampus
        "entorhinal_cortex": (130, 150, 200, 220),  # Example coordinates for entorhinal cortex
    }
    
    brain_img = np.array(image)
    
    if stage == "Mild Demented":
        cv2.rectangle(brain_img, brain_regions["hippocampus"][:2], brain_regions["hippocampus"][2:], (0, 255, 0), 2)
    elif stage == "Moderate Demented":
        cv2.rectangle(brain_img, brain_regions["hippocampus"][:2], brain_regions["hippocampus"][2:], (255, 0, 0), 2)
        cv2.rectangle(brain_img, brain_regions["entorhinal_cortex"][:2], brain_regions["entorhinal_cortex"][2:], (255, 0, 0), 2)
    elif stage == "Severe Demented":
        cv2.rectangle(brain_img, brain_regions["hippocampus"][:2], brain_regions["hippocampus"][2:], (0, 0, 255), 2)
        cv2.rectangle(brain_img, brain_regions["entorhinal_cortex"][:2], brain_regions["entorhinal_cortex"][2:], (0, 0, 255), 2)

    return brain_img

if option == "Upload MRI":
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # ensures 3 channels
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img = image.resize((128, 128))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)

        predictions = model.predict(img_array)
        class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
        predicted_class = class_names[np.argmax(predictions)]

        st.subheader("Prediction Result")
        st.success(f"The model predicts: {predicted_class}")

        # Tips Section
        st.subheader("Tips for Managing Alzheimer's (Based on Stage)")
        tips = {
            "Non Demented": "No signs of Alzheimer's. \n Maintain a healthy lifestyle: regular exercise, balanced diet, social engagement, and mental stimulation.",
            "Very Mild Demented": "Encourage daily routines, mental exercises, and regular checkups. Limit multitasking and reduce stress.",
            "Mild Demented": "Use memory aids (calendars, notes), maintain consistent routines, and offer emotional support.",
            "Moderate Demented": "Supervised care, structured environments, simplify tasks, and ensure safety at home."
        }
        advice = tips[predicted_class]
        st.info(advice)

        # Prepare PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Alzheimer's Detection Report", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"Prediction: {predicted_class}\n\nAdvice: {advice}")

        # Add uploaded image to PDF
        image_path = "uploaded_mri_temp.jpg"
        image.save(image_path)
        pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=100)

        pdf_output = io.BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)


        base64_pdf = base64.b64encode(pdf_output.read()).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="alzheimers_report.pdf">📄 Download Report as PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

elif option == "Symptom Questionnaire":
    st.subheader("📝 Alzheimer's Symptom Questionnaire")
    questions = [
        "Do you often forget recent events or conversations?",
        "Do you find it hard to follow conversations or TV shows?",
        "Do you lose track of dates, seasons, or time?",
        "Do you misplace items and can't retrace steps to find them?",
        "Do you experience confusion with words or speaking?",
        "Do you withdraw from social activities or hobbies?",
        "Do you feel changes in mood or personality?"
    ]

    score = 0
    for idx, question in enumerate(questions):
        response = st.radio(question, ["No", "Sometimes", "Yes"], key=f"q_{idx}")
        if response == "Yes":
            score += 2
        elif response == "Sometimes":
            score += 1

    if st.button("Submit Questionnaire"):
        st.subheader("🧠 Symptom-Based Risk Assessment")
        if score >= 12:
            st.error("High Risk: Please consult a neurologist or memory specialist.")
        elif score >= 6:
            st.warning("Moderate Risk: Watch symptoms closely and consider a checkup.")
        else:
            st.success("Low Risk: No immediate concerns, but maintain healthy habits.")

elif option == "Hospital Locator":
    st.subheader("🏥 Find Nearby Hospitals Treating Alzheimer's")
    city = st.text_input("Enter your city or town to search for hospitals:")

    if city:
        geolocator = Nominatim(user_agent="alzheimers_app")
        location = geolocator.geocode(city)

        if location:
            latitude = location.latitude
            longitude = location.longitude

            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json];
            node
              [amenity=hospital]
              (around:10000,{latitude},{longitude});
            out;
            """
            try:
                response = requests.get(overpass_url, params={'data': query})
                response.raise_for_status()
                data = response.json()

                if 'elements' in data:
                    map_osm = folium.Map(location=[latitude, longitude], zoom_start=12)
                    for element in data['elements']:
                        if 'lat' in element and 'lon' in element:
                            name = element.get('tags', {}).get('name', 'Unnamed Hospital')
                            folium.Marker([element['lat'], element['lon']], popup=name).add_to(map_osm)

                    st.write(f"Hospitals near **{city}**:")
                    folium_static(map_osm)
                else:
                    st.warning("No hospitals found in this area.")

            except Exception as e:
                st.error(f"Error retrieving hospital data: {e}")
        else:
            st.error("Could not find the location. Please check the city name.")

elif option == "Explain Alzheimer's Affected Regions":
    uploaded_file = st.file_uploader("Upload MRI Image to Explain Affected Regions", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image.resize((128, 128)))
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)

        predictions = model.predict(img_array)
        class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
        predicted_class = class_names[np.argmax(predictions)]

        # Map prediction to Alzheimer's stage
        if predicted_class == "Non Demented":
            stage = "Non Demented"
        elif predicted_class == "Very Mild Demented":
            stage = "Mild Demented"
        elif predicted_class == "Mild Demented":
            stage = "Moderate Demented"
        else:
            stage = "Severe Demented"

        # Show the affected brain regions based on the prediction stage
        brain_img = show_brain_regions(image, stage)
        
        # Display the highlighted brain regions
        st.image(brain_img, caption="Highlighted Affected Regions", use_column_width=True)
        
        # Prediction and explanation
        st.subheader("Prediction Result")
        st.success(f"The model predicts: {predicted_class}")
        st.subheader("Explanation")
        st.write(f"The model predicts that Alzheimer's")

elif option == "Hospital Locator":
    st.subheader("🏥 Find Nearby Hospitals Treating Alzheimer's")
    city = st.text_input("Enter your city or town to search for hospitals:")

    if city:
        geolocator = Nominatim(user_agent="alzheimers_app")
        location = geolocator.geocode(city)

        if location:
            latitude = location.latitude
            longitude = location.longitude

            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json];
            node
              [amenity=hospital]
              (around:10000,{latitude},{longitude});
            out;
            """
            try:
                response = requests.get(overpass_url, params={'data': query})
                response.raise_for_status()
                data = response.json()

                if 'elements' in data:
                    map_osm = folium.Map(location=[latitude, longitude], zoom_start=12)
                    for element in data['elements']:
                        if 'lat' in element and 'lon' in element:
                            name = element.get('tags', {}).get('name', 'Unnamed Hospital')
                            folium.Marker([element['lat'], element['lon']], popup=name).add_to(map_osm)

                    st.write(f"Hospitals near **{city}**:")
                    folium_static(map_osm)
                else:
                    st.warning("No hospitals found in this area.")

            except Exception as e:
                st.error(f"Error retrieving hospital data: {e}")
        else:
            st.error("Could not find the location. Please check the city name.")
