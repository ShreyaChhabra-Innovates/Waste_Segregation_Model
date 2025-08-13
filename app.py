import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import os

st.set_page_config(
    page_title="Waste Segregation App",
    page_icon="♻️",
    layout="centered"
)

# IMPORTANT: Replace this URL with the actual link to your MobileNetV2 model file
MODEL_URL = 'https://github.com/your-username/your-repo/releases/download/v1.0.0/mobilenet_v2_model.pth'
MODEL_PATH = 'mobilenet_v2_model.pth'

# --- Function to download the model ---
@st.cache_data
def download_model(url, path):
    """Downloads the model file from a URL if it doesn't already exist."""
    if not os.path.exists(path):
        with st.spinner("Downloading model... this may take a moment!"):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for bad responses
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download model: {e}")
                return None
    return path
# --- Load the model based on your function ---
@st.cache_resource
def load_model(model_path):
    """Load trained MobileNetV2 model with CPU/GPU handling"""
    # Ensure the model file has been downloaded before trying to load
    if not os.path.exists(model_path):
        st.error("Model file not found. Please check the download URL.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the MobileNetV2 model
    model = models.mobilenet_v2(weights=None)
    
    # Modify the classifier for a single output (binary classification)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 1),
        nn.Sigmoid()
    )
    
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        model.eval()
        st.success("Model loaded successfully!")
        return model.to(device)
    except Exception as e:
        st.error(f"Error loading model state dictionary: {e}")
        st.info("Please ensure the model file is a valid MobileNetV2 checkpoint for binary classification.")
        return None
    # --- Prediction function based on your provided logic ---
def predict(model, image):
    """Make prediction on a single image"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    device = next(model.parameters()).device
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    prob = torch.sigmoid(output).item()
    pred = "Biodegradable" if prob < 0.5 else "Non-Biodegradable"
    conf = round(100 * (1 - prob if pred == "Biodegradable" else prob), 2)
    
    return pred, conf
def main():
    """Main function to run the Streamlit app"""
    st.title('Waste Segregation Model ♻️')
    st.markdown("""
    This application uses a trained **MobileNetV2** model to classify images as either
    **Biodegradable** or **Non-Biodegradable**.
    """)

    # Download and load the model once
    model_file = download_model(MODEL_URL, MODEL_PATH)
    if model_file:
        model = load_model(model_file)
    else:
        model = None
    
    if model is None:
        st.warning("Model could not be loaded. Please check the URL and your file.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Make a prediction
        prediction, confidence = predict(model, image)

        # Display results
        st.image(image, caption='Successfully Uploaded Image', use_column_width=True)
        st.markdown(f"**Prediction:** This is **{prediction}** waste.")
        st.markdown(f"**Confidence:** The model is **{confidence:.2f}%** confident in this prediction.")

if __name__ == "__main__":
    main()
