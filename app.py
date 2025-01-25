import streamlit as st
from transformers import pipeline
import torch
import random

# Set page config first
st.set_page_config(page_title="Unlimited Text Converter", layout="wide")

# Enhanced UI
st.title("🔁 Unlimited AI to Human Text Converter")
st.markdown("Convert any length of text instantly")

# Load robust model
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return pipeline(
            "text2text-generation",
            model="t5-base",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# Advanced conversion with length preservation
def convert_text(text):
    try:
        result = model(
            f"paraphrase: {text}",
            max_length=1024,
            min_length=50,
            num_return_sequences=1,
            temperature=0.85,
            repetition_penalty=2.2,
            do_sample=True,
            truncation=True
        )
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
        return text

# Main interface
input_text = st.text_area("Input Text:", height=200,
                        placeholder="Paste any text here...")

if st.button("Convert Text") and input_text:
    with st.spinner("Converting..."):
        try:
            output = convert_text(input_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text")
                st.write(input_text)
            with col2:
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
