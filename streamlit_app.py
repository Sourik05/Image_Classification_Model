# streamlit_app.py
import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# Configure the app
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="wide"
)

# App title and description
st.title("Image Classification App")
st.markdown("""
This application allows you to upload an image and classify it using a deep learning model
trained on the Intel Image Classification dataset. The model can recognize 6 different 
natural scenes: **Buildings, Forest, Glacier, Mountain, Sea, and Street**.
""")

# API connection settings
API_URL = st.sidebar.text_input("API URL", "https:your-fastapi-deployment-url/predict")
USERNAME = st.sidebar.text_input("Username", "admin")
PASSWORD = st.sidebar.text_input("Password", "password123", type="password")

# Function to predict image class
def predict_image(image):
    if not API_URL or not USERNAME or not PASSWORD:
        st.error("❌ Please provide API URL, username, and password")
        return None

    try:
        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Send request to API
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        with st.spinner("🔍 Analyzing image..."):
            start_time = time.time()
            response = requests.post(
                API_URL,
                files=files,
                auth=(USERNAME, PASSWORD)
            )
            inference_time = time.time() - start_time

        # Check if request was successful
        if response.status_code == 200:
            return response.json(), inference_time
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Error connecting to API: {str(e)}")
        return None

# Function to plot prediction results
def plot_prediction_results(predictions):
    labels = [p["class_name"].capitalize() for p in predictions["predictions"]]
    scores = [p["confidence"] * 100 for p in predictions["predictions"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, scores, color='skyblue')

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{scores[i]:.1f}%", va='center')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Prediction Results')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    top_idx = scores.index(max(scores))
    bars[top_idx].set_color('royalblue')

    return fig

# Main app function
def main():
    st.sidebar.header("📷 Upload or Take a Photo")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Camera input
    camera_input = st.camera_input("Or take a photo with your camera 📷")
    
    # Example images
    st.sidebar.markdown("### Try with example images")
    example_categories = ["Building", "Forest", "Glacier", "Mountain", "Sea", "Street"]
    selected_example = st.sidebar.selectbox("Select example category", example_categories)

    use_example = st.sidebar.button("Use Example Image")

    image = None

    # Determine image source
    if use_example:
        example_path = f"examples/{selected_example.lower()}.jpg"
        try:
            image = Image.open(example_path)
            st.info(f"✅ Using example image: {selected_example}")
        except FileNotFoundError:
            st.error(f"❌ Example image not found: {example_path}")

    elif uploaded_file:
        image = Image.open(uploaded_file)
        st.info("✅ Using uploaded image.")

    elif camera_input:
        image = Image.open(camera_input)
        st.info("✅ Using camera input.")

    # Display image and make prediction
    if image:
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)
        
        # Predict
        predictions, inference_time = predict_image(image)
        
        if predictions:
            st.success(f"🎯 Top Prediction: **{predictions['top_prediction']}**")
            st.write(f"⏱️ Inference Time: {inference_time:.2f} sec")
            
            # Display confidence scores
            fig = plot_prediction_results(predictions)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
