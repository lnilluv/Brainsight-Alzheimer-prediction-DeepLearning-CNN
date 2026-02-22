import streamlit as st
import warnings
from PIL import Image
import requests

warnings.filterwarnings('ignore')
st.set_option('deprecation.showfileUploaderEncoding', False)

# Add a select box to choose between Alzheimer and brain tumors
options = {"Alzheimer": "https://api.brainsight.tech/predictionalz", "Brain Tumors": "https://api.brainsight.tech/predictionbt"}
selected_option = st.sidebar.radio("Select a condition to predict", options.keys())

# Upload an image and set some options for demo purposes
img_file = st.sidebar.file_uploader(label='Upload MRI scan file', type=['png', 'jpg'])

if img_file:
    img = Image.open(img_file)
    st.image(img, caption='Uploaded scan', use_column_width=True)


def on_submit():
    """Consume deep learning model through API."""
    if img_file:
        img = img_file.getvalue()
        files = {"file": img}

        endpoint = options[selected_option]
        response = requests.post(endpoint, files=files, timeout=30)
        response.raise_for_status()
        prediction = response.json()
        st.header(f"Prediction: {prediction['predicted_label']}")


st.sidebar.button('Start prediction', key=None, help=None,
                  on_click=on_submit, disabled=False)

