import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
import random
import torch
import os

# Set page config FIRST - this must be the first Streamlit command
st.set_page_config(page_title="Text Humanizer", layout="wide")

# Initialize NLTK with multiple fallback attempts
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    # Add multiple possible NLTK data paths
    nltk.data.path.append(os.path.expanduser("~/nltk_data"))
    nltk.data.path.append("/usr/share/nltk_data")
    nltk.data.path.append("/usr/local/share/nltk_data")
    nltk.data.path.append("/usr/lib/nltk_data")
    nltk.data.path.append("/usr/local/lib/nltk_data")

# Call the initialization function
initialize_nltk()

# Load model with error handling
@st.cache_resource
def load_paraphraser():
    try:
        model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

try:
    paraphraser = load_paraphraser()
except:
    st.stop()

def enhance_human_likeness(text):
    try:
        # Attempt tokenization with fallback
        try:
            words = nltk.word_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            words = nltk.word_tokenize(text)
        
        enhanced_words = []
        
        for word in words:
            try:
                if random.random() < 0.3:
                    synonyms = wordnet.synsets(word)
                    if synonyms:
                        new_word = synonyms[0].lemmas()[0].name()
                        enhanced_words.append(new_word)
                        continue
            except:
                pass
            enhanced_words.append(word)
        
        enhanced_text = ' '.join(enhanced_words)
        
        # Attempt sentence tokenization with fallback
        try:
            sentences = nltk.sent_tokenize(enhanced_text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(enhanced_text)
        
        modified_sentences = []
        for sentence in sentences:
            if random.random() < 0.4 and sentence.startswith(('The', 'It', 'This')):
                sentence = sentence[0].lower() + sentence[1:]
            modified_sentences.append(sentence)
        
        return ' '.join(modified_sentences)
    
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return text

# Streamlit UI
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
        with st.spinner("Processing... (This may take 20-40 seconds)"):
            try:
                paraphrased = paraphraser(
                    input_text,
                    max_length=len(input_text) * 2,
                    num_beams=5,
                    repetition_penalty=2.5,
                    early_stopping=True
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
    else:
        st.warning("Please input some text to process")
