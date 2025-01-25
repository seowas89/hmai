import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import random
import time
import torch
import re
from streamlit.components.v1 import html

# Set page config
st.set_page_config(page_title="Word-Preserving Humanizer", layout="wide", page_icon="✍️")

# Custom CSS
st.markdown("""
<style>
@keyframes progress {
    0% { width: 0%; }
    100% { width: 100%; }
}

.progress-bar {
    height: 4px;
    background: #4ecdc4;
    width: 100%;
    position: relative;
    animation: progress 2s linear infinite;
}

.word-count {
    color: #4ecdc4;
    font-weight: bold;
    margin: 10px 0;
}

.captcha-box {
    border: 2px solid #4ecdc4;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

# CAPTCHA system
def generate_captcha():
    if 'captcha' not in st.session_state:
        nums = [random.randint(1, 9) for _ in range(3)]
        st.session_state.captcha = {
            'question': f"{nums[0]} + {nums[1]} × {nums[2]}",
            'answer': nums[0] + nums[1] * nums[2]
        }

# Progress animation
def show_progress():
    html('<div class="progress-bar"></div>')

# Load model with word count preservation
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

def preserve_word_count(original, generated):
    original_words = original.split()
    generated_words = generated.split()
    
    # Add padding if generated is shorter
    while len(generated_words) < len(original_words):
        generated_words.append(random.choice(original_words))
    
    # Trim if generated is longer
    return ' '.join(generated_words[:len(original_words)])

def enhance_human_likeness(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    modified = []
    
    for sentence in sentences:
        words = sentence.split()
        # Preserve word count while modifying
        for i in range(len(words)):
            # Simple modifications that don't affect word count
            if random.random() < 0.3:
                words[i] = words[i].lower() if random.choice([True, False]) else words[i].upper()
            if random.random() < 0.2:
                words[i] = words[i] + random.choice([',', ';', ''])
        modified.append(' '.join(words))
    
    return ' '.join(modified)

# Generate CAPTCHA
generate_captcha()

# UI Components
st.title("🔠 Word-Preserving Humanizer")
st.markdown("Maintains exact word count while humanizing AI text")

with st.expander("🔐 CAPTCHA Verification", expanded=True):
    st.markdown(f"**Solve:** {st.session_state.captcha['question']}")
    captcha_input = st.number_input("Enter answer:", step=1)
    
    if captcha_input and int(captcha_input) != st.session_state.captcha['answer']:
        st.error("CAPTCHA verification failed. Please try again.")
        st.stop()

input_text = st.text_area("Input Text:", height=300, 
                        placeholder="Paste AI-generated text here (minimum 150 words)...")

if input_text:
    word_count = len(input_text.split())
    st.markdown(f"<div class='word-count'>Input Word Count: {word_count}</div>", 
               unsafe_allow_html=True)

if st.button("🔄 Humanize Text", type="primary") and input_text:
    if len(input_text.split()) < 150:
        st.warning("Minimum 150 words required")
        st.stop()
    
    with st.spinner("Processing..."):
        show_progress()
        start_time = time.time()
        
        try:
            # Generate with length control
            paraphrased = paraphraser(
                input_text,
                max_new_tokens=len(input_text.split()) * 2,
                min_new_tokens=len(input_text.split()),
                temperature=0.7,
                repetition_penalty=2.5,
                num_beams=3
            )[0]['generated_text']
            
            # Post-process for word count preservation
            final_output = preserve_word_count(input_text, enhance_human_likeness(paraphrased))
            
            processing_time = time.time() - start_time
            
            st.markdown(f"<div class='word-count'>Output Word Count: {len(final_output.split())}</div>", 
                       unsafe_allow_html=True)
            
            st.success("✅ Processing Complete!")
            st.text_area("Humanized Text:", value=final_output, height=300)
            
            st.download_button(
                "📥 Download Result",
                data=final_output,
                file_name="humanized_text.txt",
                mime="text/plain"
            )
            
            # Refresh CAPTCHA
            del st.session_state.captcha
            generate_captcha()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
