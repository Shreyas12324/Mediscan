import streamlit as st
import requests
import pandas as pd

# Class Labels
class_labels = {0: "Cataract", 1: "Diabetic Retinopathy", 2: "Glaucoma", 3: "Normal"}

st.title("MediScan - Eye Disease Detection")

# Initialize session state to store predictions
if "records" not in st.session_state:
    st.session_state.records = []

# User Input Fields
id_ = st.text_input("Patient ID")
name = st.text_input("Patient Name")
age = st.number_input("Patient Age", min_value=0, max_value=120, step=1)

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=200)  # 👈 Image is now smaller (width=200)

    # Ensure ID, Name, and Age are entered
    if id_ and name and age:
        files = {"file": uploaded_file.getvalue()}
        data = {"id": id_, "name": name, "age": age}

        response = requests.post("http://127.0.0.1:5000/predict", files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction")

            if prediction is not None and prediction in class_labels:
                disease = class_labels[prediction]
                st.write(f"🩺 **Predicted Disease:** {disease}")

                # Store result in session state
                st.session_state.records.append({"ID": id_, "Name": name, "Age": age, "Disease": disease})
            else:
                st.error("Invalid prediction received from backend!")

        else:
            st.error("Error in backend prediction!")

    else:
        st.warning("Please enter Patient ID, Name, and Age before uploading an image.")

# Display records as a table
if st.session_state.records:
    st.write("### Patient Records")
    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df)
