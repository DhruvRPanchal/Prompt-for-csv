import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini

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
    
    # We added an extra setting here to make sure charts work perfectly on the cloud
    df = SmartDataframe(df, config={"llm": llm, "enable_cache": False})

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating Response..."):
                # We use st.write to display text or charts automatically
                response = df.chat(prompt)
                st.write(response)
        else:
            st.warning("Enter Prompt")
