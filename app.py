import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import random
import time
import torch
import re
from streamlit.components.v1 import html

# Set page config
st.set_page_config(page_title="Length-Preserving Humanizer", layout="wide", page_icon="📏")

# Custom CSS
st.markdown("""
<style>
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.length-warning {
    color: #ff6b6b;
    font-weight: bold;
    animation: pulse 2s infinite;
}

.progress-container {
    height: 4px;
    background: #f0f0f0;
    margin: 15px 0;
    border-radius: 2px;
}

.progress-bar {
    height: 100%;
    background: #4ecdc4;
    width: 0%;
    transition: width 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# CAPTCHA system
def generate_captcha():
    if 'captcha' not in st.session_state:
        equation = f"{random.randint(5,10)} + {random.randint(1,5)}"
        answer = eval(equation)
        st.session_state.captcha = {'question': equation, 'answer': answer}

# Progress animation
def update_progress(percent):
    html(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {percent}%"></div>
    </div>
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

def preserve_length(original, generated):
    original_words = original.split()
    generated_words = generated.split()
    min_length = int(len(original_words) * 0.8)
    
    if len(generated_words) < min_length:
        needed = min_length - len(generated_words)
        generated_words += random.sample(original_words, min(needed, len(original_words)))
    
    return ' '.join(generated_words[:len(original_words)])

def enhance_human_likeness(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    modified = []
    
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            if random.random() < 0.25:
                words[i] = words[i].lower() if random.random() < 0.5 else words[i].upper()
            if random.random() < 0.1:
                words[i] += random.choice([',', ';', ''])
        modified.append(' '.join(words))
    
    return ' '.join(modified)

# Generate CAPTCHA
generate_captcha()

# UI Components
st.title("📏 Length-Preserving Text Humanizer")
st.markdown("Maintains **minimum 80%** of original text length")

with st.expander("🔒 CAPTCHA Verification", expanded=True):
    captcha_input = st.text_input(f"Solve: {st.session_state.captcha['question']} = ?", 
                                placeholder="Enter answer to verify you're human")
    if captcha_input:
        if str(st.session_state.captcha['answer']) != captcha_input.strip():
            st.error("CAPTCHA verification failed")
            st.stop()

input_text = st.text_area("Input Text:", height=300, 
                        placeholder="Paste AI-generated text here (minimum 200 words)...")

if input_text:
    original_length = len(input_text.split())
    st.markdown(f"**Original Length:** {original_length} words")
    st.markdown(f"**Minimum Target Length:** {int(original_length * 0.8)} words")

if st.button("🔄 Humanize Text", type="primary") and input_text:
    if len(input_text.split()) < 200:
        st.warning("Minimum 200 words required")
        st.stop()
    
    progress = st.empty()
    start_time = time.time()
    
    try:
        # Generate with length preservation
        update_progress(20)
        paraphrased = paraphraser(
            input_text,
            max_new_tokens=int(len(input_text.split()) * 1.2),
            min_new_tokens=int(len(input_text.split()) * 0.8),
            temperature=0.6,
            repetition_penalty=2.0,
            num_beams=4,
            do_sample=True
        )[0]['generated_text']
        
        update_progress(60)
        final_output = preserve_length(input_text, enhance_human_likeness(paraphrased))
        
        update_progress(90)
        processing_time = time.time() - start_time
        
        # Length validation
        final_length = len(final_output.split())
        length_ratio = final_length / original_length
        
        st.markdown(f"**Final Length:** {final_length} words")
        if length_ratio < 0.8:
            st.markdown("<div class='length-warning'>Warning: Could not maintain minimum length threshold</div>", 
                       unsafe_allow_html=True)
        
        st.text_area("Humanized Text:", value=final_output, height=300)
        
        st.download_button(
            "📥 Download Result",
            data=final_output,
            file_name="humanized_text.txt",
            mime="text/plain"
        )
        
        update_progress(100)
        time.sleep(0.5)
        generate_captcha()
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
