import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
import random
import torch
import os

# Set page config FIRST
st.set_page_config(page_title="Text Humanizer", layout="wide")

# Configure NLTK data path
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Initialize NLTK with robust error handling
def initialize_nltk():
    resources = {
        'punkt': ('tokenizers/punkt', ['punkt']),
        'punkt_tab': ('tokenizers/punkt_tab', ['PY3', 'english.pickle']),
        'wordnet': ('corpora/wordnet', ['wordnet'])
    }

    for resource, (path, files) in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                st.warning(f"Downloading {resource}...")
                nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)
                nltk.data.path.append(NLTK_DATA_PATH)
                # Verify download
                if not all(nltk.data.find(f"{path}/{f}") for f in files):
                    raise LookupError
            except Exception as e:
                st.error(f"Failed to download {resource}: {str(e)}")
                raise

initialize_nltk()

# Load model
@st.cache_resource
def load_paraphraser():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=512
    )

try:
    paraphraser = load_paraphraser()
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# Enhanced text processing with fallbacks
def enhance_human_likeness(text):
    try:
        # Sentence tokenization with multiple fallbacks
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
            sentences = nltk.sent_tokenize(text)
            
        modified_sentences = []
        for sentence in sentences:
            # Word tokenization with retry
            try:
                words = nltk.word_tokenize(sentence)
            except LookupError:
                nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=True)
                words = nltk.word_tokenize(sentence)
            
            # Synonym replacement
            modified_words = []
            for word in words:
                if random.random() < 0.3:
                    try:
                        synonyms = wordnet.synsets(word)
                        if synonyms:
                            new_word = synonyms[0].lemmas()[0].name()
                            modified_words.append(new_word)
                            continue
                    except:
                        pass
                modified_words.append(word)
            
            # Sentence restructuring
            modified_sentence = ' '.join(modified_words)
            if modified_sentence and modified_sentence[0].isupper():
                modified_sentence = modified_sentence[0].lower() + modified_sentence[1:]
            modified_sentences.append(modified_sentence)
            
        return ' '.join(modified_sentences)
    
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return text

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
    with st.spinner("Processing... (20-60 seconds)"):
        try:
            paraphrased = paraphraser(
                input_text,
                num_return_sequences=1,
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
