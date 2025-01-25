import streamlit as st
from transformers import pipeline
import torch
import random

# Set page config first
st.set_page_config(page_title="AI Text Converter", layout="wide")

# Simple UI with error display
st.title("🔁 AI to Human Text Converter")
st.markdown("Paste your AI-generated text below for humanization")

# Load model with error handling
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return pipeline(
            "text2text-generation",
            model="t5-small",
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# Conversion function
def convert_text(text):
    try:
        # Basic text processing
        result = model(
            f"paraphrase: {text}",
            max_length=len(text) + 50,
            num_return_sequences=1,
            temperature=0.7,
            repetition_penalty=2.5
        )
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
        return text

# Main interface
input_text = st.text_area("Input Text:", height=200,
                        placeholder="Paste your text here... (Minimum 50 words)")

if st.button("Convert Text") and input_text:
    if len(input_text.split()) < 50:
        st.warning("Please enter at least 50 words")
        st.stop()
    
    with st.spinner("Converting..."):
        try:
            output = convert_text(input_text)
            st.subheader("Converted Text")
            st.write(output)
            
            st.download_button(
                "Download Result",
                data=output,
                file_name="converted_text.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
