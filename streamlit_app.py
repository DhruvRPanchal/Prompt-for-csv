import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini  # Uses the working Gemini model
import matplotlib

# Tells the server to generate charts silently in the background
matplotlib.use('Agg') 

# Get your secret API key from Streamlit Cloud
API_KEY = st.secrets["GOOGLE_PALM2"]

# Setup the modern Gemini model
llm = GoogleGemini(api_key=API_KEY, model="gemini-1.5-flash")

st.title('Upload Your CSV File!!')
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    prompt = st.text_area("Enter Your Prompt")
    df = SmartDataframe(df, config={"llm": llm})

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating Response..."):
                st.write(df.chat(prompt))
        else:
            st.warning("Enter Prompt")
