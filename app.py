import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from parse_report import extract_text_from_pdf
from rag_engine import setup_rag

# Load environment variables (Gemini API key)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()

st.set_page_config(page_title="🏥 Medical Report Explainer")
st.title("🏥 Medical Report Explainer")
st.write("Upload your medical report (PDF) and ask questions in simple language.")

uploaded_file = st.file_uploader("📄 Upload your medical report", type=["pdf"])

report_text = ""

if uploaded_file is not None:
    # Use tempfile to avoid persistent files on Streamlit Cloud
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Extract text from PDF
    report_text = extract_text_from_pdf(temp_path)

    # Remove temp file after extraction
    os.remove(temp_path)

    # Limit extracted text size to prevent performance issues
    MAX_CHARS = 5000
    if len(report_text) > MAX_CHARS:
        st.warning(f"⚠️ Report is long; only the first {MAX_CHARS} characters will be used.")
        report_text = report_text[:MAX_CHARS]

    st.subheader("📝 Extracted Report Text")
    st.text_area("Here's the extracted content from your PDF:", report_text, height=300)

@st.cache_resource(show_spinner="Loading retrieval engine...")
def load_rag_chain():
    return setup_rag()

rag_chain = load_rag_chain()

if report_text:
    st.subheader("💬 Ask Questions About Your Report")

    user_question = st.text_input("What would you like to know?", "")

    if user_question:
        with st.spinner("Thinking..."):
            full_prompt = user_question + "\n\nReport:\n" + report_text
            response = rag_chain.run(full_prompt)

        st.markdown("### 🤖 AI Explanation:")
        st.write(response)
