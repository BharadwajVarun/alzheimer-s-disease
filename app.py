from fpdf import FPDF
import datetime
import streamlit as st

def step_upload_and_predict():
    import tensorflow as tf
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    st.header("Step 1: Upload MRI and Predict Alzheimer's Stage")
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

        def preprocess_image(img):
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            return np.expand_dims(img_array, axis=0)

        def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, tf.argmax(predictions[0])]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()

        def overlay_heatmap(image, heatmap):
            heatmap = cv2.resize(heatmap, (image.width, image.height))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
            return Image.fromarray(superimposed_img)

        try:
            model = tf.keras.models.load_model("your_model.h5")  # replace with your model
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

            st.success(f"Predicted Stage: **{class_names[predicted_class]}**")

            # Grad-CAM explanation
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv1')  # change if needed
            highlighted_img = overlay_heatmap(image, heatmap)
            st.image(highlighted_img, caption="Highlighted Alzheimer's Region", use_column_width=True)

            # Store in session for later
            st.session_state["prediction_result"] = class_names[predicted_class]
            st.session_state["highlighted_img"] = highlighted_img
            st.session_state["mri_uploaded"] = True

        except Exception as e:
            st.error(f"Prediction or explanation error: {e}")
    else:
        st.info("Please upload an MRI image.")

import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
from fpdf import FPDF
import base64
import io

# Set page configuration (MUST BE FIRST)
st.set_page_config(page_title="Alzheimer's Detection", page_icon="🧠", layout="wide")

# Load model with caching
@st.cache_resource
def load_alzheimer_model():
    model = load_model('best_finetuned_model.h5')
    return model

model = load_alzheimer_model()

# Initialize session state if not already done
if 'step' not in st.session_state:
    st.session_state.step = 1  # Start at Step 1
    st.session_state.report_data = {}  # Initialize empty dictionary for report data
    st.session_state.predicted_class = None  # Initialize the predicted class
    st.session_state.uploaded_image = None  # Initialize uploaded image as None
    st.session_state.symptom_score = None  # Initialize symptom score

if "mri_image" not in st.session_state:
    st.session_state.mri_image = None

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

def generate_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    return heatmap

def find_last_conv_layer(model):
    # Return the name of the last 2D convolutional layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


# Function to update session state with collected data
def update_session_data(key, value):
    st.session_state.report_data[key] = value

# Sidebar for step navigation
def sidebar_navigation():
    st.sidebar.title("Navigation")
    steps = {
        1: "Upload MRI Image",
        2: "Symptom Questionnaire",
        3: "Hospital Locator",
        4: "Explain Alzheimer's Affected Regions",
        5: "Generate Full Report"
    }
    st.session_state.step = st.sidebar.radio("Select Step", list(steps.keys()), format_func=lambda x: steps[x])

    if steps == "Tips & Guidance":
        st.title("🧠 Alzheimer's Prevention & Care Tips")

        st.subheader("Prevention Tips")
        st.markdown("""
    - Stay physically and mentally active (e.g., walking, chess, puzzles).
    - Maintain a balanced diet rich in fruits, vegetables, and omega-3s.
    - Control chronic diseases like hypertension and diabetes.
    - Sleep 7–8 hours per night for brain recovery.
    - Avoid smoking and limit alcohol consumption.
        """)

        st.subheader("Caregiver Tips")
        st.markdown("""
    - Create a structured routine for the patient.
    - Use memory aids like sticky notes or digital reminders.
    - Be calm, patient, and avoid correcting harshly.
    - Take care of your own mental and physical well-being.
    - Join caregiver support groups for advice and relief.
        """)

    
# Call sidebar navigation at the top
sidebar_navigation()

# Step 1: Upload MRI
if st.session_state.step == 1:
    st.header("Step 1: Upload MRI Image")
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.mri_image = image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image and make prediction
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        predictions = model.predict(img_array)
        class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
        predicted_class = class_names[np.argmax(predictions)]

        st.subheader("Prediction Result")
        st.success(f"The model predicts: {predicted_class}")

        # Save the uploaded image and prediction result to session state
        st.session_state.uploaded_image = uploaded_file
        st.session_state.predicted_class = predicted_class
        # Save MRI image and model to session state
        st.session_state.mri_image = image  # mri_image should be the image object you processed
        st.session_state.model = model  # model should be the loaded/trained model


# Step 2: Symptom Questionnaire
elif st.session_state.step == 2:
    st.header("Step 2: Alzheimer's Symptom Questionnaire")
    if st.session_state.predicted_class is None:
        st.warning("Please upload an MRI image first.")
    else:
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
            
            # Save the questionnaire score to session state
            st.session_state.symptom_score = score

# Step 3: Hospital Locator
elif st.session_state.step == 3:
    #st.header("Step 3: Find Nearby Hospitals Treating Alzheimer's")
    if st.session_state.predicted_class is None:
        st.warning("Please upload an MRI image first.")
    else:
        import streamlit as st
        from geopy.geocoders import Nominatim
        import overpy

# Initialize OSM API and geolocator
        api = overpy.Overpass()
        geolocator = Nominatim(user_agent="alzheimers_locator")

# Get user location
        location = st.text_input("Enter your location (city or area):")

# Alzheimer’s-related keywords
        alz_keywords = ['neuro', 'memory', 'brain', 'alzheimer', 'dementia']

        if location:
            try:
                loc = geolocator.geocode(location)
                if loc:
                    lat, lon = loc.latitude, loc.longitude

            # Overpass query to get hospitals around location
                    radius = 5000  # in meters
                    query = f"""
            [out:json];
            node["amenity"="hospital"](around:{radius},{lat},{lon});
            out body;
            """
                    result = api.query(query)

            # Filter hospitals
                    filtered = []
                    for node in result.nodes:
                        name = node.tags.get("name", "Unnamed Hospital")
                        if any(keyword in name.lower() for keyword in alz_keywords):
                            filtered.append({
                        "name": name,
                        "lat": node.lat,
                        "lon": node.lon,
                        "address": f"{node.lat}, {node.lon}"  # Can use reverse geocode for full address
                    })

            # Show results
                    if filtered:
                        st.success("Found the following hospitals possibly treating Alzheimer's:")
                        for i, hosp in enumerate(filtered, start=1):
                            st.markdown(f"**{i}. {hosp['name']}**")
                            st.markdown(f"- 📍 Location: `{hosp['address']}`")
                    else:
                        st.warning("No Alzheimer-specific hospitals found. Here are general hospitals instead:")
                        for i, node in enumerate(result.nodes, start=1):
                            name = node.tags.get("name", "Unnamed Hospital")
                            st.markdown(f"**{i}. {name}**")
                            st.markdown(f"- 📍 Location: `{node.lat}, {node.lon}`")
                else:
                    st.error("Couldn't locate the place. Try a more specific address.")

            except Exception as e:
                st.error(f"Error: {e}")





# Step 4: Explain Alzheimer's Affected Regions
elif st.session_state.step == 4:

    import streamlit as st
    from PIL import Image
    import matplotlib.pyplot as plt

    if "mri_image" in st.session_state and "model" in st.session_state:
        model = st.session_state.model
        mri_image = st.session_state.mri_image
        st.subheader("🧠 Explainable AI - Affected Brain Region")

        image = st.session_state["mri_image"]
        model = st.session_state["model"]

        img = image.resize((224, 224))  # Match your model input size
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    # ✅ Replace this with your actual last conv layer name
        last_conv_layer_name = find_last_conv_layer(model)  # Use your model's last conv layer name here

        try:
            last_conv_layer_name = find_last_conv_layer(model)
            heatmap = generate_gradcam(model, img_array, last_conv_layer_name)

            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(np.array(img), 0.7, heatmap, 0.3, 0)

            st.image(superimposed_img, caption="Highlighted Alzheimer’s region", use_column_width=True)

        except Exception as e:
            st.error(f"Error generating Grad-CAM: {e}")


    else:
        st.warning("MRI image or model not found. Please upload and predict first.")



# Step 5: Generate Report (PDF)
elif st.session_state.step == 5:
    st.header("Step 5: Generate and Download Full Report")
    if st.session_state.predicted_class is None:
        st.warning("Please upload an MRI image first.")
    else:
        import pandas as pd
        hospital_df = pd.DataFrame({
    'name': ['Hospital A', 'Hospital B', 'Hospital C'],
    'lat': [12.9716, 13.0827, 13.0674],
    'lon': [77.5946, 80.2707, 80.2370],
    'tags': ['neurology, alzheimer', 'neurology', 'alzheimer']
})
        filtered_hospitals = hospital_df[hospital_df['tags'].str.contains('neurology|alzheimer', case=False, na=False)]
        hospital_list = [
            {"name": row['name'], "lat": row['lat'], "lon": row['lon']}
            for _, row in filtered_hospitals.iterrows()
        ]
        hospital_df = pd.DataFrame(hospital_list)
        
        def generate_pdf_report(predicted_class, advice, mri_image=None):
    # Create PDF object
            pdf = FPDF()
            pdf.add_page()

    # Title
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Alzheimer's Detection Report", ln=True, align='C')
            pdf.ln(10)

    # Prediction and Advice Section
            pdf.multi_cell(0, 10, txt=f"Prediction: {predicted_class}\n\nAdvice: {advice}")

    # If MRI Image is provided, add it to the PDF
            if mri_image:
        # Save the uploaded image temporarily
                image_path = "uploaded_mri_temp.jpg"
                mri_image.save(image_path)
                pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=100)

    # Prepare PDF to be returned as a downloadable file
            pdf_output = io.BytesIO()
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0)

    # Convert PDF to base64 for download link
            base64_pdf = base64.b64encode(pdf_output.read()).decode('utf-8')

            return base64_pdf

# Streamlit UI for uploading MRI image
        st.title("Alzheimer's Risk Detection")

# User inputs
        prediction = st.selectbox("Select Alzheimer's Risk Prediction", ["High Risk", "Low Risk"])
        advice = st.text_area("Advice and Recommendations", "Enter your advice here based on the risk.")

# Upload MRI image
        mri_image = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

        if st.button("Generate PDF Report"):
    # Generate the PDF report
            base64_pdf = generate_pdf_report(predicted_class=prediction, advice=advice, mri_image=Image.open(mri_image) if mri_image else None)
    
    # Create a download link for the PDF
            href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="alzheimers_report.pdf">📄 Download Report as PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
    # Tips Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Helpful Tips & Guidance:", ln=True)
            pdf.set_font("Arial", '', 12)
    
    # Prevention Tips
            pdf.multi_cell(0, 10, "🧠 Prevention Tips:")
            pdf.multi_cell(0, 10, "- Stay physically and mentally active (walking, puzzles).")
            pdf.multi_cell(0, 10, "- Maintain a healthy diet rich in fruits and omega-3.")
            pdf.multi_cell(0, 10, "- Manage blood pressure, diabetes, and cholesterol.")
            pdf.multi_cell(0, 10, "- Get enough quality sleep.")
            pdf.multi_cell(0, 10, "- Avoid smoking and limit alcohol.")
            pdf.ln(5)

    # Caregiving Tips
            pdf.multi_cell(0, 10, "🤝 Caregiver Tips:")
            pdf.multi_cell(0, 10, "- Establish a routine for daily tasks.")
            pdf.multi_cell(0, 10, "- Use memory aids (labels, sticky notes).")
            pdf.multi_cell(0, 10, "- Be patient and calm with communication.")
            pdf.multi_cell(0, 10, "- Take care of your own health and seek support groups.")
            pdf.multi_cell(0, 10, "- Consult professionals when needed.")

    # Save PDF
           
        
           
            href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="./alzheimers_report.pdf">📄 Download Report as PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
