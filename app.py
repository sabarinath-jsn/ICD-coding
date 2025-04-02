# Workaround for Streamlit's torch compatibility issue
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
from torch import nn
import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
import json
import requests
import io
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Radiology Report ICD Code Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Drive file IDs and URLs
MODEL_ARTIFACTS = {
    "category_mlb.pkl": "11Raeps5stBjKp1H_cFAgj5AhZqM9Hmrq",
    "mlb.pkl": "1dwM6SP5sXKwYMSnBciiibyy86Cy8bZBO",
    "model_config.json": "15InuT50L9QcCycPbzL7GmBI2DJBI87F0",
    "model_weights.pth": "1rplrueYVMkOYxt17I5K2bcq7gaN__0lI",
    "special_tokens_map.json": "1sa-iNEaE5Ods-TRbscr4R0_RzM1yf1oE",
    "tokenizer_config.json": "1R9ayyMiqvFBr30jDiRimXBG7iXQEMcQ-",
    "tokenizer.json": "1sfLCU6XlOMVQ6Jtf9YZ6Q91CYC5-eISV",
    "vocab.txt": "1CHLG9i_l9MLziNQAm3OqnWFE0xGw6HJS"
}

# Function to download files from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Download all model artifacts
@st.cache_resource
def download_model_artifacts():
    artifact_dir = Path("model_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    
    for filename, file_id in MODEL_ARTIFACTS.items():
        dest_path = artifact_dir / filename
        if not dest_path.exists():
            with st.spinner(f"Downloading {filename}..."):
                download_file_from_google_drive(file_id, dest_path)
    
    return artifact_dir

# Load spaCy model for preprocessing
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load("en_core_web_sm")
    ts = termset("en_clinical")
    nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})
    return nlp

nlp = load_spacy_model()

# Define the model architecture (must match training)
class HierarchicalAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_categories, num_heads=4, dropout=0.2):
        super(HierarchicalAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.category_attention_heads = nn.ModuleList(
            [nn.Linear(input_dim, 1, bias=False) for _ in range(num_heads)]
        )
        self.subcategory_attention_heads = nn.ModuleList(
            [nn.Linear(input_dim, 1, bias=False) for _ in range(num_heads)]
        )
        self.category_dropout = nn.Dropout(dropout)
        self.subcategory_dropout = nn.Dropout(dropout)
        self.category_norm = nn.LayerNorm(input_dim)
        self.subcategory_norm = nn.LayerNorm(input_dim)
        self.category_projection = nn.Linear(input_dim, num_categories, bias=False)
        self.subcategory_projection = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(1)
        category_scores = torch.mean(torch.stack(
            [torch.softmax(head(x), dim=1) for head in self.category_attention_heads], dim=-1)
        category_scores = self.category_dropout(category_scores)
        category_representation = torch.sum(x * category_scores, dim=1)
        category_representation = self.category_norm(category_representation + torch.mean(x, dim=1))
        category_representation = self.category_projection(category_representation)
        
        subcategory_scores = torch.mean(torch.stack(
            [torch.softmax(head(x), dim=1) for head in self.subcategory_attention_heads], dim=-1)
        subcategory_scores = self.subcategory_dropout(subcategory_scores)
        subcategory_representation = torch.sum(x * subcategory_scores, dim=1)
        subcategory_representation = self.subcategory_norm(
            subcategory_representation + torch.mean(x, dim=1))
        subcategory_representation = self.subcategory_projection(subcategory_representation)
        
        combined_representation = torch.cat([category_representation, subcategory_representation], dim=-1)
        return combined_representation, category_scores, subcategory_scores

class ICDClassifierWithHierarchicalAttention(nn.Module):
    def __init__(self, input_dim, num_categories, num_subcategories):
        super(ICDClassifierWithHierarchicalAttention, self).__init__()
        self.hierarchical_attention = HierarchicalAttentionLayer(input_dim, num_categories, num_heads=8, dropout=0.3)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim + num_categories, 2048), nn.ReLU(), nn.Dropout(0.5), nn.LayerNorm(2048))
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5), nn.LayerNorm(1024))
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.LayerNorm(512))
        self.fc4 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.LayerNorm(256))
        self.residual_projection1 = nn.Linear(2048, 1024)
        self.residual_projection2 = nn.Linear(1024, 512)
        self.residual_projection3 = nn.Linear(512, 256)
        self.output_layer = nn.Linear(256, num_subcategories)

    def forward(self, x):
        combined_representation, category_scores, subcategory_scores = self.hierarchical_attention(x)
        x = self.fc1(combined_representation)
        residual1 = x
        x = self.fc2(x)
        x += self.residual_projection1(residual1)
        residual2 = x
        x = self.fc3(x)
        x += self.residual_projection2(residual2)
        residual3 = x
        x = self.fc4(x)
        x += self.residual_projection3(residual3)
        output = self.output_layer(x)
        return output, category_scores, subcategory_scores

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    # Download artifacts first
    artifact_dir = download_model_artifacts()
    
    # Load config
    with open(artifact_dir / 'model_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = ICDClassifierWithHierarchicalAttention(
        input_dim=config['input_dim'],
        num_categories=config['num_categories'],
        num_subcategories=config['num_subcategories']
    )
    
    # Load weights
    model.load_state_dict(torch.load(
        artifact_dir / 'model_weights.pth',
        map_location=torch.device('cpu')
    ))
    model.eval()
    
    # Load label encoders
    mlb = joblib.load(artifact_dir / 'mlb.pkl')
    category_mlb = joblib.load(artifact_dir / 'category_mlb.pkl')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
    
    return model, mlb, category_mlb, tokenizer

model, mlb, category_mlb, tokenizer = load_model_artifacts()

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())
    negated_tokens = []
    for ent in doc.ents:
        if ent._.negex:
            negated_tokens.append("NOT_" + ent.text)
        else:
            negated_tokens.append(ent.text)
    return ' '.join(negated_tokens)

# Get embeddings function
def get_embeddings(texts, batch_size=4, max_length=512):
    # Ensure input is a list
    if isinstance(texts, str): texts = [texts]
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

# Prediction function
def predict_icd_codes(text, threshold=0.5):
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Get embeddings
    embeddings = get_embeddings([processed_text])
    
    # Convert to tensor
    input_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        outputs, _, _ = model(input_tensor)
        probs = torch.sigmoid(outputs).numpy()[0]
    
    # Apply threshold
    predictions = (probs > threshold).astype(int)
    
    # Get predicted ICD codes
    predicted_codes = mlb.inverse_transform(np.array([predictions]))
    
    # Get top probabilities
    top_indices = np.argsort(probs)[::-1][:5]  # Top 5 predictions
    top_codes = mlb.classes_[top_indices]
    top_probs = probs[top_indices]
    
    return predicted_codes[0], list(zip(top_codes, top_probs))

# LIME explanation function
def explain_prediction(text, icd_code, index):
    explainer = LimeTextExplainer(class_names=mlb.classes_)
    
    def predict_proba(texts):
        processed_texts = [preprocess_text(t) for t in texts]
        embeddings = get_embeddings(processed_texts)
        input_tensor = torch.tensor(embeddings, dtype=torch.float32)
        with torch.no_grad():
            outputs, _, _ = model(input_tensor)
            return torch.sigmoid(outputs).numpy()
    
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,
        labels=[index]
    )
    
    return exp.as_html()

# Streamlit UI
def main():
    st.title("🏥 Radiology Report ICD Code Predictor")
    st.markdown("""
    This tool predicts ICD-10 codes from radiology reports using a deep learning model with hierarchical attention.
    """)
    
    # Input section
    with st.expander("📝 Enter Radiology Report", expanded=True):
        report_text = st.text_area(
            "Paste the radiology report text here:",
            height=300,
            placeholder="CT scan of the abdomen and pelvis with contrast...",
            help="Enter the full radiology report text for ICD code prediction"
        )
    
    # Threshold slider
    threshold = st.slider(
        "Prediction Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust the sensitivity of the predictions. Higher values mean more confident predictions."
    )
    
    # Prediction button
    if st.button("Predict ICD Codes", disabled=not report_text):
        if not report_text.strip():
            st.warning("Please enter a radiology report")
            return
        
        with st.spinner("Analyzing report and predicting ICD codes..."):
            # Make prediction
            predicted_codes, top_predictions = predict_icd_codes(report_text, threshold)
            
            # Display results
            st.subheader("Prediction Results")
            
            if not predicted_codes:
                st.warning("No ICD codes predicted above the selected threshold")
            else:
                st.success(f"Predicted ICD Codes: {', '.join(predicted_codes)}")
            
            # Show top predictions in a table
            st.markdown("### Top Predictions")
            top_df = pd.DataFrame(top_predictions, columns=["ICD Code", "Probability"])
            st.dataframe(top_df.style.format({"Probability": "{:.2%}"}), width=800)
            
            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Probability Distribution")
                fig, ax = plt.subplots()
                ax.barh(top_df["ICD Code"], top_df["Probability"])
                ax.set_xlabel("Probability")
                ax.set_title("Top Predicted ICD Codes")
                st.pyplot(fig)
            
            # LIME explanation for top prediction
            if top_predictions:
                top_code = top_predictions[0][0]
                code_index = np.where(mlb.classes_ == top_code)[0][0]
                
                with st.spinner(f"Generating explanation for {top_code}..."):
                    lime_html = explain_prediction(report_text, top_code, code_index)
                    st.markdown("### Prediction Explanation", unsafe_allow_html=True)
                    st.components.v1.html(lime_html, height=500, scrolling=True)

# Run the app
if __name__ == "__main__":
    import pandas as pd  # Import moved here to avoid caching issues
    main()
