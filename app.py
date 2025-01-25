import streamlit as st
from transformers import pipeline, AutoTokenizer
import torch
import random

# Set page config first
st.set_page_config(page_title="Exact-Length Text Converter", layout="wide")

st.title("📏 Exact-Length AI to Human Converter")
st.markdown("Maintains **identical word count** during conversion")

# Load model and tokenizer
@st.cache_resource
def load_components():
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = pipeline(
            "text2text-generation",
            model="t5-base",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()

model, tokenizer = load_components()

def exact_length_conversion(text):
    # Calculate exact token length
    inputs = tokenizer(text, return_tensors="pt").input_ids
    exact_length = inputs.shape[1]
    
    # Generate with tight length control
    result = model(
        f"paraphrase: {text}",
        max_length=exact_length + 2,
        min_length=exact_length - 2,
        num_beams=5,
        temperature=0.4,
        repetition_penalty=2.5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )[0]['generated_text']
    
    # Post-process to match exact word count
    original_words = text.split()
    result_words = result.split()
    
    # Pad or truncate to match exact length
    if len(result_words) < len(original_words):
        needed = len(original_words) - len(result_words)
        result_words += random.sample(original_words, needed)
    elif len(result_words) > len(original_words):
        result_words = result_words[:len(original_words)]
    
    return ' '.join(result_words)

# UI components
input_text = st.text_area("Input Text:", height=200,
                        placeholder="Paste text to convert (any length)...")

if st.button("Convert Text") and input_text:
    with st.spinner("Converting while preserving length..."):
        try:
            original_word_count = len(input_text.split())
            output = exact_length_conversion(input_text)
            converted_word_count = len(output.split())
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Original ({original_word_count} words)")
                st.write(input_text)
            with col2:
                st.subheader(f"Converted ({converted_word_count} words)")
                st.write(output)
            
            st.download_button(
                "Download Exact-Length Result",
                data=output,
                file_name="converted_text.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Conversion error: {str(e)}")
