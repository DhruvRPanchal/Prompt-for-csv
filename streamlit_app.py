import streamlit as st
import pandas as pd
from google import genai

# 1. Dashboard Layout Setup
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title('📊 Upload Your CSV File!')

# 2. Extract your secure API key from Streamlit Cloud Secrets
if "GOOGLE_PALM2" in st.secrets:
    API_KEY = st.secrets["GOOGLE_PALM2"]
else:
    st.error("Missing API Key! Please verify GOOGLE_PALM2 is added to your Streamlit App Secrets.")
    st.stop()

# 3. Setup the official, reliable Google Gemini Client
client = genai.Client(api_key=API_KEY)

# 4. File Uploader in Sidebar
uploaded_file = st.sidebar.file_uploader("Upload your CSV data here:", type="csv")

if uploaded_file:
    # Read the data file safely
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview (First 5 Rows)")
    st.write(df.head())

    # User Prompt Input Box
    st.subheader("💬 Ask your Data a Question")
    prompt = st.text_area("Example: 'What is the average sales?' or 'Summarize the trend in this data'")
    
    # Run the Generation Action
    if st.button("Generate Answer", type="primary"):
        if prompt:
            with st.spinner("Gemini is reading your dataset... Please wait..."):
                try:
                    # Package the data structure to safely hand over to Gemini 
                    data_summary = df.to_string(max_rows=20)
                    
                    full_instruction = f"""
                    You are an expert data analyst. Below is a snapshot of the user's uploaded CSV data:
                    
                    {data_summary}
                    
                    The user is asking the following question about this data: "{prompt}"
                    Please look closely at the rows and columns provided above and write a helpful, mathematically accurate answer.
                    """
                    
                    # Generate the reply using the official Gemini model
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=full_instruction,
                    )
                    
                    st.success("Analysis Complete!")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(f"An error occurred while communicating with Gemini: {e}")
        else:
            st.warning("Please type a question or prompt first!")
else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to get started.")
