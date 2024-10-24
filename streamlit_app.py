import streamlit as st
from gtts import gTTS
from PyPDF2 import PdfReader
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
from io import BytesIO

# Download necessary NLTK resources
nltk.download('punkt')

# Load the model and tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text() + " "
    return full_text

# Convert text to audio
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_buffer = BytesIO()
    tts.save(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Function for file preprocessing
def preprocess_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    full_text = extract_text_from_pdf(pdf_file)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_text(full_text)
    
    return chunks

# LLM pipeline for summarization
def summarize_text(text):
    summarization_pipeline = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    
    summary = summarization_pipeline(text, truncation=True)[0]['summary_text']
    return summary

# Function to display PDF
@st.cache_data
def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit App
st.title("Document Summarization and PDF to Speech App")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:

     # Convert the extracted text to speech
    st.info("Converting PDF text to speech...")
    audio_buffer = text_to_audio(pdf_text)
    st.audio(audio_buffer, format='audio/mp3')

    # Display PDF in the app
    st.info("Uploaded PDF")
    display_pdf(uploaded_file)

    # Extract text from the PDF
    st.info("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Summarize the extracted text
    st.info("Generating summary...")
    summary = summarize_text(pdf_text)
    st.success("Summarization Complete")
    st.write(summary)
