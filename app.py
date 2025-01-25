# app.py
import streamlit as st
from transformers import pipeline
import torch

# Set page config
st.set_page_config(page_title="Free AI Text Humanizer", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        max-width: 800px;
        padding: 2rem;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .humanize-btn {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white !important;
        width: 100%;
        padding: 1rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    textarea {
        border: 2px solid #4CAF50 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("Free AI Text Humanizer")
st.markdown("Turn AI-generated text into natural, human-like writing")
st.markdown('</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_humanizer():
    return pipeline(
        "text2text-generation",
        model="humarin/chatgpt_paraphraser_on_T5_base",
        device=0 if torch.cuda.is_available() else -1
    )

humanizer = load_humanizer()

# Input text area
input_text = st.text_area(
    "Paste your text here", 
    height=250,
    placeholder="Enter AI-generated text to humanize..."
)

# Humanization button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("Humanize Text", key="humanize_btn"):
        if input_text.strip():
            with st.spinner("Humanizing..."):
                try:
                    result = humanizer(
                        input_text,
                        max_length=len(input_text) + 100,
                        num_beams=5,
                        temperature=0.7,
                        repetition_penalty=2.5
                    )[0]['generated_text']
                    
                    st.subheader("Humanized Text")
                    st.markdown(f"""
                    <div style="
                        background: #f5f5f5;
                        padding: 1.5rem;
                        border-radius: 8px;
                        margin-top: 1rem;
                    ">
                        {result}
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
        else:
            st.warning("Please enter some text to humanize")
