import streamlit as st
from gtts import gTTS
from PyPDF2 import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import base64
import os

# Load LaMini-Flan-T5-248M from Hugging Face
@st.cache_resource
def load_lamini_model():
    # Load tokenizer and model from Hugging Face
    tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    model = T5ForConditionalGeneration.from_pretrained("MBZUAI/LaMini-Flan-T5-248M", device_map="auto", torch_dtype="auto")
    
    # Load summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

# Convert PDF to text
def pdf_to_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)

    # Extract text from all pages
    full_text = ""
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        full_text += page.extract_text() + " "

    return full_text

# Convert text to audio using gTTS
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)
    return audio_file

# Summarize text using the preloaded model pipeline
def summarize_text(text):
    summarizer = load_lamini_model()
    
    # LaMini model has a token limit, summarize in chunks if necessary
    chunk_size = 1024  # Define chunk size within model limits
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    summarized_text = ""
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summarized_text += summary[0]['summary_text'] + " "

    return summarized_text

# Display the PDF in Streamlit
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app
st.title("PDF Summarizer and Text to Speech Converter")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    filepath = "temp_" + uploaded_file.name
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    
    st.info("Extracting text from PDF...")
    pdf_text = pdf_to_text(filepath)
    
    #Convert text to audio
    st.info("Converting text to audio...")
    audio_file = text_to_audio(pdf_text)
    st.audio(audio_file, format='audio/mp3')

    # Summarize the extracted text
    st.info("Summarizing the document...")
    summary = summarize_text(pdf_text)
    st.success("Summarization complete!")
    st.write(summary)

    # Convert the summary to audio
    st.info("Converting summary to audio...")
    audio_file = text_to_audio(summary)
    st.audio(audio_file, format='audio/mp3')
    
    # Display the PDF and summary side by side
    col1, col2 = st.columns(2)

    with col1:
        st.info("Uploaded PDF")
        displayPDF(filepath)

    with col2:
        st.info("Summary of PDF")
        st.write(summary)

    # Cleanup temp files (optional)
    os.remove(filepath)
    os.remove(audio_file)
    
