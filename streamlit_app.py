import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini

# 1. Page Configuration
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title('📊 Upload Your CSV File!')

# 2. Securely pull the Google Gemini API Key from Streamlit Dashboard Secrets
if "GOOGLE_PALM2" in st.secrets:
    API_KEY = st.secrets["GOOGLE_PALM2"]
else:
    st.error("Missing API Key! Please add GOOGLE_PALM2 to your Streamlit Secrets.")
    st.stop()

# 3. Initialize the modern Gemini AI Model
llm = GoogleGemini(api_key=API_KEY, model="gemini-1.5-flash")

# 4. Sidebar File Uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV data here:", type="csv")

if uploaded_file:
    # Read and show preview of data
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview (First 5 Rows)")
    st.write(df.head())

    # User question input
    st.subheader("💬 Ask your Data a Question")
    prompt = st.text_area("Example: 'What is the average sales?' or 'Make a bar chart of categories'")
    
    # Configure PandasAI with custom chart paths for cloud environments
    df_smart = SmartDataframe(df, config={
        "llm": llm, 
        "enable_cache": False,
        "save_charts": True,
        "save_charts_path": "/tmp"
    })

    # Action Button
    if st.button("Generate Answer", type="primary"):
        if prompt:
            with st.spinner("Analyzing your data... Please wait..."):
                try:
                    response = df_smart.chat(prompt)
                    st.success("Analysis Complete!")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
        else:
            st.warning("Please type a question or prompt first!")
else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to get started.")
