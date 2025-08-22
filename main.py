import streamlit as st

st.set_page_config(page_title="PetroNus: Well Log Analysis Platform", layout="centered")

# Enhanced CSS for professional look with larger content and background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #d0c7b6 0%, #978f86 100%);
        background-attachment: fixed;
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem auto;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    .stButton > button {
        height: 120px;
        font-size: 1.4rem;
        font-weight: 600;
        border-radius: 12px;
        border: 1px solid #ddd;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #495057 0%, #343a40 100%);
        border-color: #343a40;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    h1 {
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 1rem;
        color: #2c3e50;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .subtitle {
        font-size: 2rem !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem !important;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        color: #2c3e50;
    }
    /* Remove link icons from headings */
    .stMarkdown h1 a,
    .stMarkdown h2 a,
    .stMarkdown h3 a,
    .stMarkdown h4 a,
    .stMarkdown h5 a,
    .stMarkdown h6 a {
        display: none !important;
    }
    .stMarkdown h1:hover a,
    .stMarkdown h2:hover a,
    .stMarkdown h3:hover a,
    .stMarkdown h4:hover a,
    .stMarkdown h5:hover a,
    .stMarkdown h6:hover a {
        display: none !important;
    }
    /* Remove any anchor links */
    a[href^="#"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("PetroNus")
st.markdown('<p class="subtitle"><strong>A Petrophysical Parameter Estimation and Payzone Detection Platform</strong></p>', unsafe_allow_html=True)

st.markdown("---")

st.markdown('<h3 class="section-header">Select Analysis Tool</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Payzone Predictor", use_container_width=True):
        st.switch_page("pages/1_Payzone_Pred.py")

with col2:
    if st.button("Phi & Sw Predictor", use_container_width=True):
        st.switch_page("pages/2_Phi_Sw_Pred.py")

st.markdown("---")