"""
ResearchIQ – Streamlit UI
Run with: streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="ResearchIQ", page_icon="🔬", layout="wide")
st.title("🔬 ResearchIQ – Multimodal AI Research Assistant")
st.caption("Hackathon prototype – work in progress")

# --- Sidebar: paper upload ---
with st.sidebar:
    st.header("📄 Papers")
    uploaded = st.file_uploader("Upload PDF papers", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        st.success(f"{len(uploaded)} file(s) ready to ingest")
        if st.button("Ingest papers"):
            st.info("Ingestion pipeline not yet implemented.")

# --- Tabs ---
tab_qa, tab_compare, tab_synthesis, tab_visual = st.tabs(
    ["💬 Q&A", "🔁 Compare", "📚 Synthesis", "🖼️ Visual Q&A"]
)

with tab_qa:
    st.subheader("Ask a question across the paper collection")
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        st.info("QA chain not yet implemented.")

with tab_compare:
    st.subheader("Compare two papers")
    col1, col2 = st.columns(2)
    with col1:
        paper_a = st.selectbox("Paper A", options=["(none)"])
    with col2:
        paper_b = st.selectbox("Paper B", options=["(none)"])
    aspect = st.selectbox("Aspect", ["methodology", "results", "architecture", "datasets", "limitations"])
    if st.button("Compare"):
        st.info("Comparison chain not yet implemented.")

with tab_synthesis:
    st.subheader("Literature synthesis across all papers")
    if st.button("Generate synthesis"):
        st.info("Synthesis chain not yet implemented.")

with tab_visual:
    st.subheader("Ask about a figure, chart, or table")
    fig_question = st.text_input("Question about a visual element")
    if st.button("Ask about figure") and fig_question:
        st.info("Visual chain not yet implemented.")
