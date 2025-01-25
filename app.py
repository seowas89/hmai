import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import random
import torch

# Set page config FIRST
st.set_page_config(page_title="Text Humanizer", layout="wide")

# Load spaCy model
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

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
    doc = nlp(text)
    modified_sentences = []
    
    for sent in doc.sents:
        words = [token.text for token in sent]
        # Random word modifications
        for i in range(len(words)):
            if random.random() < 0.3:
                words[i] = words[i].lower() if random.random() < 0.5 else words[i].upper()
        
        # Sentence structure variation
        modified = " ".join(words)
        if modified and modified[0].isupper():
            modified = modified[0].lower() + modified[1:]
        modified_sentences.append(modified)
    
    return " ".join(modified_sentences)

# Streamlit UI
st.title("AI to Human Text Converter")

with st.expander("⚠️ Important Disclaimer"):
    st.write("""
    - Use ethically and verify outputs
    - No guarantee against AI detection
    - For educational purposes only
    """)

input_text = st.text_area("Paste AI-generated text here:", height=250)

if st.button("Humanize Text") and input_text:
    with st.spinner("Processing..."):
        try:
            paraphrased = paraphraser(
                input_text,
                max_length=512,
                num_beams=5,
                temperature=0.7,
                repetition_penalty=2.5
            )[0]['generated_text']
            
            final_output = enhance_human_likeness(paraphrased)
            
            st.subheader("Humanized Output")
            st.code(final_output, language="text")
            
            st.download_button(
                label="Download Result",
                data=final_output,
                file_name="humanized_text.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
