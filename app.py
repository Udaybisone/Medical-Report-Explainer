import streamlit as st
import os
from dotenv import load_dotenv
from parse_report import extract_text_from_pdf
from rag_engine import setup_rag

# Load environment variables (Gemini API key)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="🏥 Medical Report Explainer")
st.title("🏥 Medical Report Explainer")
st.write("Upload your medical report (PDF) and ask questions in simple language.")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your medical report", type=["pdf"])

report_text = ""

if uploaded_file is not None:
    # Save file to disk temporarily
    with open("temp_report.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract text from PDF
    report_text = extract_text_from_pdf("temp_report.pdf")

    st.subheader("📝 Extracted Report Text")
    st.text_area("Here's the extracted content from your PDF:", report_text, height=300)

# Ask questions about the report


if report_text:
    st.subheader("💬 Ask Questions About Your Report")

    user_question = st.text_input("What would you like to know?", "")

    if user_question:
        with st.spinner("Thinking..."):
            rag_chain = setup_rag()
            full_prompt = user_question + "\n\nReport:\n" + report_text
            response = rag_chain.run(full_prompt)

        st.markdown("### 🤖 AI Explanation:")
        st.write(response)
