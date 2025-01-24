# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
import random
import torch

# Initialize NLTK
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Load model with explicit tokenizer configuration
@st.cache_resource
def load_paraphraser():
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # Force slow tokenizer
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

# Streamlit UI
st.set_page_config(page_title="Text Humanizer", layout="wide")

# ... [Keep the rest of the Streamlit UI code from previous version] ...
