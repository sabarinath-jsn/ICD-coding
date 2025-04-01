import streamlit as st
import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModel
import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
import gdown
import os

# --- Constants ---
GOOGLE_DRIVE_FOLDER = "1EnM2_BAKVmGq--3B4b2Os7z5Z-_fR-hq"
MODEL_FILES = {
    "mlb.pkl": "1ABC123",  # Replace with actual file IDs
    "category_mlb.pkl": "1XYZ456",
    "icd_model.pth": "1DEF789"
}

# --- Helper Functions ---
def download_from_gdrive(file_id, output):
    """Download a file from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

def ensure_models_downloaded():
    """Ensure all model files are downloaded"""
    os.makedirs("models", exist_ok=True)
    for filename, file_id in MODEL_FILES.items():
        path = f"models/{filename}"
        if not os.path.exists(path):
            with st.spinner(f"Downloading {filename}..."):
                download_from_gdrive(file_id, path)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    ensure_models_downloaded()
    
    # Load tokenizer and BioClinicalBERT
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    # Load spaCy with negation
    nlp = spacy.load("en_core_web_sm")
    ts = termset("en_clinical")
    nlp.add_pipe("negex", config={"neg_termset": ts.get_patterns()})
    
    # Load label encoders
    mlb = joblib.load("models/mlb.pkl")
    category_mlb = joblib.load("models/category_mlb.pkl")
    
    # Load ICD classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icd_model = torch.load("models/icd_model.pth", map_location=device)
    icd_model.eval()
    
    return tokenizer, bert_model, nlp, mlb, category_mlb, icd_model

# --- Text Processing ---
def preprocess_text(text, nlp):
    doc = nlp(text.lower())
    processed_tokens = []
    for ent in doc.ents:
        prefix = "NOT_" if ent._.negex else ""
        processed_tokens.append(f"{prefix}{ent.text}")
    return " ".join(processed_tokens)

def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# --- Prediction Function ---
def predict_icd_codes(text, tokenizer, bert_model, nlp, mlb, icd_model):
    processed_text = preprocess_text(text, nlp)
    embeddings = get_embeddings(processed_text, tokenizer, bert_model)
    
    with torch.no_grad():
        inputs = torch.tensor(embeddings).float()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs, _, _ = icd_model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    threshold = 0.5
    predictions = [(mlb.classes_[i], float(probs[i])) 
                 for i in np.where(probs > threshold)[0]]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions, processed_text

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Radiology ICD Coder", layout="wide")
    
    st.title("📝 Radiology ICD Coding Assistant")
    st.markdown("""
    Paste a radiology report below to automatically predict ICD-10 codes.
    """)
    
    report_text = st.text_area("**Input Radiology Report**", height=300,
                             placeholder="Paste radiology report text here...")
    
    if st.button("🚀 Analyze Report", type="primary"):
        if not report_text.strip():
            st.error("Please enter a radiology report")
            return
        
        with st.spinner("🧠 Processing report..."):
            try:
                # Load all models
                tokenizer, bert_model, nlp, mlb, category_mlb, icd_model = load_models()
                
                # Get predictions
                predictions, processed_text = predict_icd_codes(
                    report_text, tokenizer, bert_model, nlp, mlb, icd_model
                )
                
                # Display results
                st.subheader("🔍 Predicted ICD Codes")
                if not predictions:
                    st.warning("No ICD codes predicted with high confidence.")
                else:
                    cols = st.columns(2)
                    for i, (code, prob) in enumerate(predictions):
                        with cols[i % 2]:
                            st.metric(
                                label=f"**{code}**",
                                value=f"{prob:.1%} confidence",
                                help=f"Probability: {prob:.4f}"
                            )
                
                with st.expander("🔧 View processed text"):
                    st.code(processed_text)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()