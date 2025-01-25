import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import random
import torch
import re

# Set page config FIRST
st.set_page_config(page_title="Text Humanizer", layout="wide")

# Load paraphrasing model
@st.cache_resource
def load_paraphraser():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

paraphraser = load_paraphraser()

def enhance_human_likeness(text):
    # Basic sentence splitting
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    modified = []
    for sentence in sentences:
        # Simple word modifications
        words = sentence.split()
        for i in range(len(words)):
            # Random lowercase with 30% probability
            if random.random() < 0.3:
                words[i] = words[i].lower()
        # Rebuild sentence
        modified_sentence = ' '.join(words)
        # Add random comma with 20% chance
        if random.random() < 0.2 and len(modified_sentence) > 20:
            modified_sentence = modified_sentence[:-1] + ', ' + modified_sentence[-1]
        modified.append(modified_sentence)
    
    return ' '.join(modified)

# Streamlit UI
st.title("AI to Human Text Converter")

with st.expander("⚠️ Disclaimer"):
    st.write("Ethical use only. Verify outputs manually.")

input_text = st.text_area("Paste AI text here:", height=250)

if st.button("Humanize Text") and input_text:
    with st.spinner("Processing..."):
        try:
            paraphrased = paraphraser(
                input_text,
                max_length=512,
                temperature=0.9,
                repetition_penalty=2.0
            )[0]['generated_text']
            
            final_output = enhance_human_likeness(paraphrased)
            
            st.subheader("Output")
            st.write(final_output)
            
            st.download_button(
                "Download",
                data=final_output,
                file_name="humanized.txt"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
