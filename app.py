import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import random
import time
import torch
import re
from streamlit.components.v1 import html

# Set page config
st.set_page_config(page_title="Text Humanizer Pro", layout="wide", page_icon="✍️")

# Custom CSS for animations and styling
st.markdown("""
<style>
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

.progress-bar {
    height: 4px;
    background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
    animation: progress 2s ease-in-out infinite;
    margin: 10px 0;
}

@keyframes progress {
    0% { width: 0%; }
    50% { width: 100%; }
    100% { width: 0%; }
}

.captcha-box {
    border: 2px solid #4ecdc4;
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
}

.result-box {
    animation: fadeIn 0.5s ease-in;
    border-left: 5px solid #4ecdc4;
    padding: 15px;
    background: #f8f9fa;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# CAPTCHA system
def generate_captcha():
    if 'captcha' not in st.session_state:
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        st.session_state.captcha = {'question': f"{a} + {b}", 'answer': a + b}

# Progress animation
def show_progress_animation():
    html("""
    <div class="progress-bar"></div>
    """)

# Load model
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
    # Improved text preservation
    sentences = re.split(r'(?<=\?|\.|\!)\s+', text)
    modified = []
    
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            # Smarter modifications
            if random.random() < 0.2:
                words[i] = words[i].lower() if random.choice([True, False]) else words[i].upper()
            if random.random() < 0.1:
                words[i] += random.choice([',', ';', '...'])
        modified.append(' '.join(words))
    
    return ' '.join(modified)

# Generate CAPTCHA
generate_captcha()

# UI Components
st.title("📝 Text Humanizer Pro")
st.markdown("Transform AI-generated text into human-like content with enhanced preservation")

with st.expander("⚙️ Settings", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        captcha_input = st.text_input(f"Solve: {st.session_state.captcha['question']} = ?", 
                                    placeholder="Enter answer to verify you're human")
    with col2:
        st.markdown("<div style='height: 50px'></div>", unsafe_allow_html=True)
        if captcha_input:
            if str(st.session_state.captcha['answer']) != captcha_input.strip():
                st.error("❌ Incorrect CAPTCHA. Please try again.")
                st.stop()

input_text = st.text_area("Paste your AI-generated text here:", height=250,
                        placeholder="Enter text to humanize (minimum 200 characters)...")

if st.button("🚀 Humanize Text", disabled=not input_text or len(input_text) < 200):
    if len(input_text) < 200:
        st.warning("⚠️ Please enter at least 200 characters")
        st.stop()
    
    with st.spinner(""):
        show_progress_animation()
        start_time = time.time()
        
        try:
            # Improved paraphrasing parameters
            paraphrased = paraphraser(
                input_text,
                max_new_tokens=1024,
                num_beams=5,
                temperature=0.85,
                repetition_penalty=2.2,
                do_sample=True
            )[0]['generated_text']
            
            final_output = enhance_human_likeness(paraphrased)
            processing_time = time.time() - start_time
            
            st.markdown(f"<div class='result-box'>📝 **Humanized Text** (processed in {processing_time:.1f}s):</div>", 
                       unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 15px; border-radius: 5px; background: #fff;'>{final_output}</div>", 
                       unsafe_allow_html=True)
            
            st.download_button(
                "💾 Download Result",
                data=final_output,
                file_name="humanized_text.txt",
                mime="text/plain",
                type="primary"
            )
            
            # Regenerate CAPTCHA after successful processing
            del st.session_state.captcha
            generate_captcha()
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
