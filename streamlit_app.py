import streamlit as st
from gtts import gTTS
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os

# NLTK resources
nltk.download('punkt')

# Model path
checkpoint = "/mount/src/blank-app-4/LaMini-Flan-T5-248M"

# Load the model and tokenizer
if os.path.exists(checkpoint):
    try:
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
    except OSError as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model checkpoint path '{checkpoint}' does not exist.")

# Convert PDF to text
def pdf_to_speech(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)

    full_text = ""
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        full_text += page.extract_text() + " "

    return full_text 

# Convert text to audio
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)

    return audio_file 

# File preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# LLM summarization pipeline
def llm_pipeline(filepath):
    try:
        pipe_sum = pipeline(
            'summarization',
            model=base_model,
            tokenizer=tokenizer,
            max_length=500,
            min_length=50
        )
        
        input_text = file_preprocessing(filepath)
        result = pipe_sum(input_text)
        summary = result[0]['summary_text']
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

@st.cache_data
# Display the PDF
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit UI
st.title("Document Summarization and PDF to Speech App")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    
    filepath = "temp_" + uploaded_file.name
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.info("Converting PDF to speech...")
    pdf_text = pdf_to_speech(filepath)
    audio_file = text_to_audio(pdf_text)
    st.audio(audio_file, format='audio/mp3')

    col1, col2 = st.columns(2)

    with col1:
        st.info("Uploaded PDF")
        displayPDF(filepath)

    with col2:
        st.info("Generating Summary...")
        summary = llm_pipeline(filepath)
        st.success("Summarization Complete")
        st.write(summary)

    # Cleanup temp files
    os.remove(filepath)
    os.remove(audio_file)
