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
        4: "Explain Alzheimer's Affected Regions",
        3: "Hospital Locator",
        5: "Generate Full Report",
        6: "Know about Alzheimer's"
    }
    st.session_state.step = st.sidebar.radio("Select Step", list(steps.keys()), format_func=lambda x: steps[x])

    if steps == "Tips & Guidance":
        st.title(" Alzheimer's Prevention & Care Tips")

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
    
    # 🧠 Alzheimer's Info Section
    st.markdown("""
    ### 🧠 About Alzheimer's Disease
    Alzheimer's is a progressive neurological disorder that leads to memory loss, cognitive decline, and behavioral changes. 
    Early detection through MRI imaging and symptom analysis can help slow its progression and improve quality of life.
    
    This tool uses deep learning to analyze MRI scans and predict potential stages of Alzheimer's:
    - **Non Demented**: No signs of dementia.
    - **Very Mild Demented**: Early signs with minimal memory issues.
    - **Mild Demented**: Noticeable cognitive decline.
    - **Moderate Demented**: Significant impact on daily life.

    Please upload an MRI image to begin the analysis.
    """)

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

        # Save to session state
        st.session_state.uploaded_image = uploaded_file
        st.session_state.predicted_class = predicted_class
        st.session_state.mri_image = image
        st.session_state.model = model
        st.session_state["predicted_stage"] = predicted_class


# Step 2: Symptom Questionnaire
elif st.session_state.step == 2:
    st.header("Step 2: Alzheimer's Symptom Questionnaire")

    if st.session_state.predicted_class is None:
        st.warning("Please upload an MRI image first.")
    else:
        st.markdown("Please answer the following questions to assess common symptoms.")

        questions = [
            "Do you often forget recent events or conversations (Eg: forgetting names of people)?",
            "Do you find it hard to follow conversations or TV shows?",
            "Do you lose track of dates, seasons, or time?",
            "Do you misplace items and can't retrace steps to find them?",
            "Do you experience confusion with words or speaking?",
            "Do you withdraw from social activities or hobbies?",
            "Do you feel changes in mood or personality? (Eg: Feeling sad, immediate aggressive)",
            "Do you have trouble planning or solving problems?",
            "Do you feel disoriented even in familiar surroundings?",
            "Do you have trouble completing daily tasks at home or work?",
            "Have your judgment or decision-making skills declined recently?",
            "Do friends or family mention memory issues you've not noticed?"
        ]

        score = 0
        for idx, question in enumerate(questions):
            response = st.radio(question, ["No", "Sometimes", "Yes"], key=f"q_{idx}")
            if response == "Yes":
                score += 2
            elif response == "Sometimes":
                score += 1

        max_score = len(questions) * 2

        if st.button("Submit Questionnaire"):
            st.subheader("🧠 Symptom-Based Risk Assessment")
            st.progress(score / max_score)

            if score >= 20:
                st.error("High Risk: Please consult a neurologist or memory specialist.")
                st.markdown("**Tips:**")
                st.markdown("- Schedule a medical consultation.")
                st.markdown("- Keep a daily routine and journal.")
                st.markdown("- Involve a caregiver or family member in regular activities.")
            elif score >= 12:
                st.warning("Moderate Risk: Monitor your symptoms and consider a checkup.")
                st.markdown("**Tips:**")
                st.markdown("- Engage in brain-stimulating activities (e.g., puzzles, reading).")
                st.markdown("- Follow a healthy diet (e.g., Mediterranean diet).")
                st.markdown("- Exercise regularly and socialize.")
            else:
                st.success("Low Risk: No immediate concerns.")
                st.markdown("**Tips:**")
                st.markdown("- Maintain a healthy lifestyle.")
                st.markdown("- Continue engaging in learning and hobbies.")
                st.markdown("- Get regular checkups as a preventive measure.")

            st.session_state.symptom_score = score
            st.session_state["questionnaire_score"] = score




# Step 3: Hospital Locator
elif st.session_state.step == 3:
    st.header("Step 4: Find Nearby Hospitals Treating Alzheimer's")

    if st.session_state.predicted_class is None:
        st.warning("Please upload an MRI image first.")
    else:
        import streamlit as st
        from geopy.geocoders import Nominatim
        import overpy
        import folium
        from streamlit_folium import st_folium

        # Initialize OSM API and geolocator
        api = overpy.Overpass()
        geolocator = Nominatim(user_agent="alzheimers_locator")

        # Get user location input
        location = st.text_input("Enter your city or location:")

        # Keywords for Alzheimer’s-related filtering
        alz_keywords = ['alzheimer', 'dementia', 'neurology', 'memory', 'brain']

        if location:
            try:
                loc = geolocator.geocode(location)
                if loc:
                    lat, lon = loc.latitude, loc.longitude
                    radius = 8000  # in meters

                    # Overpass Query to fetch nearby hospitals
                    query = f"""
                    [out:json];
                    node["amenity"="hospital"](around:{radius},{lat},{lon});
                    out body;
                    """
                    result = api.query(query)

                    # Filter hospitals using Alzheimer's-related keywords
                    filtered = []
                    for node in result.nodes:
                        tags_combined = " ".join(node.tags.values()).lower()
                        if any(keyword in tags_combined for keyword in alz_keywords):
                            filtered.append({
                                "name": node.tags.get("name", "Unnamed Hospital"),
                                "lat": node.lat,
                                "lon": node.lon,
                            })

                    # Fallback to general hospital list if no matches found
                    if not filtered:
                        st.warning("No Alzheimer-specific hospitals found. Showing general hospitals.")
                        filtered = [{
                            "name": node.tags.get("name", "Unnamed Hospital"),
                            "lat": node.lat,
                            "lon": node.lon,
                        } for node in result.nodes]

                    # Display results
                    st.success(f"Found {len(filtered)} hospital(s):")
                    for i, hosp in enumerate(filtered, start=1):
                        st.markdown(f"**{i}. {hosp['name']}**")
                        st.markdown(f"- 📍 Location: `{hosp['lat']}, {hosp['lon']}`")

                    # Create map view using Folium
                    map_center = [lat, lon]
                    m = folium.Map(location=map_center, zoom_start=13)

                    for hosp in filtered:
                        folium.Marker(
                            location=[hosp["lat"], hosp["lon"]],
                            popup=hosp["name"],
                            icon=folium.Icon(color='blue', icon='plus-sign')
                        ).add_to(m)

                    # Display map in Streamlit
                    st_folium(m, width=700, height=500)

                else:
                    st.error("Couldn't find that location. Try a more specific city or area.")

            except Exception as e:
                st.error(f"An error occurred: {e}")






# Step 4: Explain Alzheimer's Affected Regions
elif st.session_state.step == 4:
    import numpy as np
    from PIL import Image
    import cv2
    st.subheader("🧠 Explainable AI - Affected Brain Region")

    if "mri_image" in st.session_state and "predicted_stage" in st.session_state:
        image = st.session_state["mri_image"]
        stage = st.session_state["predicted_stage"]

        def show_brain_regions(image, stage):
            brain_regions = {
                "hippocampus": (90, 140, 130, 180),
                "entorhinal_cortex": (70, 160, 110, 200),
            }

            brain_img = np.array(image.resize((224, 224)))

            if stage == "Very Mild Demented":
                cv2.rectangle(brain_img, brain_regions["hippocampus"][:2], brain_regions["hippocampus"][2:], (0, 255, 0), 2)
            elif stage == "Mild Demented":
                cv2.rectangle(brain_img, brain_regions["hippocampus"][:2], brain_regions["hippocampus"][2:], (255, 0, 0), 2)
                cv2.rectangle(brain_img, brain_regions["entorhinal_cortex"][:2], brain_regions["entorhinal_cortex"][2:], (255, 0, 0), 2)
            elif stage == "Moderate Demented":
                cv2.rectangle(brain_img, brain_regions["hippocampus"][:2], brain_regions["hippocampus"][2:], (0, 0, 255), 2)
                cv2.rectangle(brain_img, brain_regions["entorhinal_cortex"][:2], brain_regions["entorhinal_cortex"][2:], (0, 0, 255), 2)

            return brain_img

        result_img = show_brain_regions(image, stage)
        st.image(result_img, caption=f"Affected regions for stage: {stage}", use_column_width=True)
        st.session_state["highlighted_image"] = result_img

    else:
        st.warning("MRI image or predicted stage not found. Please upload and predict first.")





# Step 5: Generate Report (PDF)
elif st.session_state.step == 5:
    st.header("Step 5: Generate and Download Full Report")

    import io
    import base64
    from fpdf import FPDF

    if "predicted_class" not in st.session_state or "mri_image" not in st.session_state:
        st.warning("Please complete prediction first.")
    else:
        # Inputs from session and UI
        predicted_class = st.session_state.predicted_class
        highlighted_image = st.session_state.get("highlighted_image")  # result_img from step 4
        questionnaire_score = st.session_state.get("questionnaire_score", "Not answered")

        advice = st.text_area("Doctor's Advice or Observations", "Patient shows early signs. Recommend further clinical evaluation.")

        class AlzheimerPDF(FPDF):
            def header(self):
                self.set_font("Arial", 'B', 14)
                self.cell(0, 10, " Alzheimer's Diagnostic Report", ln=True, align='C')
                self.ln(10)

            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", 'I', 8)
                self.cell(0, 10, f"Page {self.page_no()}", align='C')

        pdf = AlzheimerPDF()
        pdf.add_page()
        pdf.add_font('DejaVu', '', r'/home/systemadministrator/Documents/varun\/dejavu-fonts-ttf-2.37/ttf/DejaVuSerifCondensed.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
        # Section: Prediction
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"Prediction Result: {predicted_class}\n\nSymptom Questionnaire Score: {questionnaire_score}")

        # Section: Advice
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Doctor's Advice:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, advice)
        pdf.ln(5)

        # Section: Highlighted MRI
        if highlighted_image is not None:
            path = "highlighted_temp.jpg"
            Image.fromarray(highlighted_image).save(path)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "MRI Scan with Affected Regions:", ln=True)
            pdf.image(path, x=10, w=100)
            pdf.ln(10)

        # Section: Tips
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Helpful Tips:", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 10, 
        """- Stay mentally and physically active
- Maintain a healthy, balanced diet
- Sleep well and manage stress
- Use memory aids (notes, labels)
- Seek professional support when needed""")

        # Generate and Download PDF
        pdf_bytes = pdf.output(dest='S').encode('latin1')  # Convert to bytes

        pdf_buffer = io.BytesIO()
        pdf_buffer.write(pdf_bytes)
        #pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        base64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')

        st.download_button(label="📄 Download Report as PDF",
                           data=base64.b64decode(base64_pdf),
                           file_name="alzheimers_report.pdf",
                           mime='application/pdf')

elif st.session_state.step == 6:
    st.title("🧠 Learn About Alzheimer's Disease")

    st.markdown("""
    ### What is Alzheimer's Disease?
    Alzheimer's is a degenerative brain disease and the most common cause of dementia. 
    It affects memory, thinking, and behavior, and symptoms worsen over time.

    ### Common Symptoms:
    - Memory loss
    - Difficulty performing familiar tasks
    - Trouble understanding visual images
    - Confusion with time or place
    - Mood or personality changes

    ### Causes and Risk Factors:
    - Age (65+ is the most common)
    - Genetics
    - Brain changes (amyloid plaques, tau tangles)
    - Head injuries
    - Lifestyle and heart health

    ### Diagnosis:
    - MRI scans to detect structural brain changes
    - Cognitive questionnaires
    - Medical history and lab tests

    ### Prevention and Management:
    - Regular exercise and a healthy diet
    - Mental stimulation (reading, puzzles)
    - Social engagement
    - Managing blood pressure and cholesterol
    - Early detection and care planning

    > Early diagnosis can help in better management and improved quality of life.
    """)
