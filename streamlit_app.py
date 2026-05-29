import streamlit as st
import pandas as pd
from google import genai
import plotly.express as px

# ==============================================================================
# 1. PAGE AND THEME CONFIGURATION (Showcases professional UI/UX design)
# ==============================================================================
st.set_page_config(
    page_title="Enterprise AI Data Analytics Workspace", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('📊 Enterprise AI Data Analytics Workspace')
st.markdown("""
    *Developed as a high-performance framework integrating native interactive visualizations with the Google Gemini 2.5 Flash Engine.*
""")

# ==============================================================================
# 2. SECURE CREDENTIAL & STATE MANAGEMENT (Demonstrates production security mindset)
# ==============================================================================
if "GOOGLE_PALM2" in st.secrets:
    API_KEY = st.secrets["GOOGLE_PALM2"]
else:
    st.error("🔒 **Security Block:** Missing API Key configuration. Please append `GOOGLE_PALM2` to your Streamlit platform secrets.")
    st.stop()

# Initialize official Gemini Client
@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)

client = get_gemini_client(API_KEY)

# ==============================================================================
# 3. APPLICATION SIDEBAR & INTERACTIVE FILTERS (Showcases advanced Pandas skills)
# ==============================================================================
st.sidebar.header("📁 Data Ingestion Pipeline")
uploaded_file = st.sidebar.file_uploader("Upload Target Dataset (CSV)", type="csv")

if uploaded_file:
    # Read data file safely
    base_df = pd.read_csv(uploaded_file)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Dynamic Runtime Filters")
    
    # Allow the user to isolate specific columns dynamically
    all_cols = base_df.columns.tolist()
    filter_col = st.sidebar.selectbox("Isolate Column for Inspection:", all_cols)
    
    # Calculate unique values to show off native business intelligence processing
    unique_vals = base_df[filter_col].nunique()
    st.sidebar.metric(label=f"Unique Items in [{filter_col}]", value=f"{unique_vals:,}")
    
    # App main dashboard layout split into production tabs
    tab1, tab2, tab3 = st.tabs(["📋 Data Explorer", "📈 Advanced Interactive Charts", "🧠 AI Cognitive Analytics"])
    
    # --------------------------------------------------------------------------
    # TAB 1: DATA EXPLORER
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("Dataset Structural Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{base_df.shape[0]:,}")
        col2.metric("Total Columns", f"{base_df.shape[1]:,}")
        col3.metric("Missing Data Cells", f"{base_df.isna().sum().sum():,}")
        
        st.subheader("Data Snapshot (First 10 Rows)")
        st.dataframe(base_df.head(10), use_container_width=True)
        
        st.subheader("Statistical Profile Summary")
        st.dataframe(base_df.describe(include='all').fillna('-'), use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: ADVANCED INTERACTIVE CHARTS (Demonstrates master visualization skill)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Interactive Business Intelligence Charts")
        
        numeric_cols = base_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = base_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                x_axis = st.selectbox("Select X-Axis Variable:", all_cols, index=0)
                y_axis = st.selectbox("Select Y-Axis Variable (Numeric):", numeric_cols, index=0)
                chart_type = st.radio("Visualization Strategy:", ["Line Chart", "Bar Chart", "Scatter Plot"], horizontal=True)
            
            with chart_col2:
                color_target = st.selectbox("Group/Color Classification By (Optional):", [None] + categorical_cols)
            
            # Execute advanced Plotly engine rendering
            st.markdown("---")
            if chart_type == "Line Chart":
                fig = px.line(base_df, x=x_axis, y=y_axis, color=color_target, title=f"{y_axis} Trend Over {x_axis}")
            elif chart_type == "Bar Chart":
                fig = px.bar(base_df, x=x_axis, y=y_axis, color=color_target, title=f"{y_axis} Distribution Across {x_axis}")
            else:
                fig = px.scatter(base_df, x=x_axis, y=y_axis, color=color_target, title=f"Correlation Matrix: {x_axis} vs {y_axis}")
                
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient continuous numeric columns identified to execute automated visual graphics pipeline.")

    # --------------------------------------------------------------------------
    # TAB 3: AI COGNITIVE ANALYTICS (Showcases optimized AI prompt engineering)
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Natural Language Data Synthesis Engine")
        prompt = st.text_area(
            "Query the dataset using natural English text:", 
            placeholder="e.g., 'Analyze the overarching macroeconomic trend and isolate the top three performing anomalies.'"
        )
        
        if st.button("Execute Cognitive Graph Call", type="primary"):
            if prompt:
                with st.spinner("Streaming operational data vectors directly into Gemini Engine..."):
                    try:
                        # Optimization: We send metadata and shape definitions to minimize token overhead costs
                        structural_summary = f"""
                        - Dataset Total Rows: {base_df.shape[0]}
                        - Dataset Total Columns: {base_df.shape[1]}
                        - Data Columns Layout & Data Types: {base_df.dtypes.to_dict()}
                        - Basic Summary Descriptors: {base_df.describe(include='all').to_string(max_rows=15)}
                        - Top Content Sample Snapshot: {base_df.head(15).to_string()}
                        """
                        
                        system_instruction = f"""
                        You are acting as an elite Principal Data Scientist and Business Intelligence Director.
                        Evaluate the structural metrics below and give highly detailed, mathematically objective, 
                        and actionable insights. Keep formatting beautifully structured with clear bold headers, tables, or lists where needed.
                        
                        [DATASET PROFILE STRUCTURE]:
                        {structural_summary}
                        
                        [EXPLICIT USER QUERY]:
                        "{prompt}"
                        """
                        
                        # Generate payload call
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=system_instruction,
                        )
                        
                        st.success("Analysis Processing Engine Successful!")
                        st.markdown("### 🧠 AI Analytics Output")
                        st.write(response.text)
                        
                    except Exception as e:
                        st.error(f"Operational Framework Exception Encountered: {e}")
            else:
                st.warning("Execution halted. Prompt buffer is completely empty.")
else:
    # Professional onboarding dashboard landing page when no data is uploaded yet
    st.info("👋 Welcome to the Enterprise Data Dashboard. Please upload a target corporate CSV dataset on the left command sidebar panel to launch the automated parsing pipelines.")
