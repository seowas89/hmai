# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
import random
import torch

# Set page config FIRST
st.set_page_config(page_title="Text Humanizer", layout="wide")

# Initialize NLTK
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Load model with explicit tokenizer configuration
@st.cache_resource
def load_paraphraser():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

paraphraser = load_paraphraser()

def enhance_human_likeness(text):
    words = nltk.word_tokenize(text)
    enhanced_words = []
    
    for word in words:
        if random.random() < 0.3:
            synonyms = wordnet.synsets(word)
            if synonyms:
                new_word = synonyms[0].lemmas()[0].name()
                enhanced_words.append(new_word)
                continue
        enhanced_words.append(word)
    
    enhanced_text = ' '.join(enhanced_words)
    sentences = nltk.sent_tokenize(enhanced_text)
    
    modified_sentences = []
    for sentence in sentences:
        if random.random() < 0.4 and sentence.startswith(('The', 'It', 'This')):
            sentence = sentence[0].lower() + sentence[1:]
        modified_sentences.append(sentence)
    
    return ' '.join(modified_sentences)

# Streamlit UI components come AFTER page config
st.title("AI to Human Text Converter")

with st.expander("⚠️ Important Disclaimer"):
    st.write("""
    - Use this tool responsibly and ethically
    - No guarantee of bypassing detection systems
    - Always comply with guidelines
    - Verify outputs manually
    """)

input_text = st.text_area("Paste AI-generated text here:", height=250)

if st.button("Humanize Text"):
    if input_text:
        with st.spinner("Processing..."):
            paraphrased = paraphraser(input_text, max_length=len(input_text))[0]['generated_text']
            final_output = enhance_human_likeness(paraphrased)
            
            st.subheader("Humanized Output")
            st.code(final_output, language="text")
            
            st.download_button(
                label="Download Result",
                data=final_output,
                file_name="humanized_text.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please input some text to process")
