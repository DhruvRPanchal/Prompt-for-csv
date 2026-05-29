import streamlit as st
import pandas as pd
from google import genai

# Simple Layout Setup
st.set_page_config(page_title="AI Data Calculator", layout="wide")
st.title('📊 Ask Questions & Calculate Data')

# Secure API Key Check
if "GOOGLE_PALM2" in st.secrets:
    API_KEY = st.secrets["GOOGLE_PALM2"]
else:
    st.error("Missing API Key! Please add GOOGLE_PALM2 to your Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type="csv")

if uploaded_file:
    # Read and show the raw table
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Your Data Preview")
    st.write(df.head(10))

    # Simple Prompt Input Box
    st.subheader("💬 What calculation or question do you have?")
    prompt = st.text_input(
        "Type here...", 
        placeholder="e.g., 'What is the total number of .. ?' or 'List total ... by ...' "
    )
    
    if st.button("Calculate", type="primary"):
        if prompt:
            with st.spinner("Calculating..."):
                try:
                    # Convert the entire data sheet to text so Gemini can read every row for math
                    full_data_string = df.to_string(index=False)
                    
                    instructions = f"""
                    You are a precise data calculation engine. 
                    Look at the dataset below and answer the user's specific question using exact math.
                    
                    DATASET:
                    {full_data_string}
                    
                    USER QUESTION:
                    "{prompt}"
                    
                    Provide a direct, accurate answer. Show your basic calculation step if necessary so it's easy to verify.
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=instructions,
                    )
                    
                    st.success("Result:")
                    st.write(response.text)
                    
                except Exception as e:
                    st.error(f"Error processing calculation: {e}")
        else:
            st.warning("Please enter a question first!")
else:
    st.info("Please upload a CSV file in the sidebar to begin.")
