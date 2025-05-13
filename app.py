import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

# Load model and processor
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224" # Model trained on Image Net
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, processor


# Prediction Function
def classify(image,model,processor):
    inputs = processor(images = image,return_tensors ="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        return model.config.id2label[predicted_class] # return human readable label
    
# Stremalit UI
st.title("Hugging Face Classification Model")
st.markdown("Uplaod Image Here")
uploaded_file = st.file_uploader("Upload an Image: ", type=['jpg','jpeg','png'])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image=image,caption="Uploaded Image",use_column_width=True)
    model,processor = load_model()
    prediction = classify(image,model,processor)
    
    st.subheader(f"Prediction: {prediction}")