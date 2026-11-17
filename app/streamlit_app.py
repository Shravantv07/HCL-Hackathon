
import streamlit as st
import requests

st.set_page_config(
    page_title="ResearchIQ",
    page_icon="🔬",
    layout="wide"
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🔬 ResearchIQ")
    st.caption("Multimodal AI Research Assistant")

    st.divider()

    st.subheader("Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Upload your research PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} papers selected")

        for file in uploaded_files:
            st.write("📄", file.name)
    else:
        st.info("Upload PDFs to get started.")

    st.divider()
    st.caption("Research Topic")
    st.write("Agentic AI & Multi-Agent Systems")


# ---------- MAIN PAGE ----------
st.title("ResearchIQ")
st.subheader("Your AI-Powered Research Companion")

st.write(
    "Analyze research papers, compare methodologies, "
    "explore figures, and discover research insights."
)

st.divider()

tab1, tab2, tab3 = st.tabs([
    "💬 Research Chat",
    "📊 Paper Comparison",
    "🖼️ Visual Analysis"
])

# ---------- CHAT TAB ----------
with tab1:
    st.header("Ask Your Research Assistant")

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            for source in message.get("sources", []):
                st.caption(
                    f"📄 {source.get('paper', 'Unknown')} "
                    f"— Page {source.get('page', '?')}"
                )

    # User enters a question
    question = st.chat_input(
        "Ask a question about your research papers..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Send question to FastAPI
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/ask",
                        json={
                            "question": question,
                            "selected_papers": []
                        },
                        timeout=60
                    )

                    response.raise_for_status()
                    result = response.json()

                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    # Display citations
                    for source in sources:
                        st.caption(
                            f"📄 {source.get('paper', 'Unknown')} "
                            f"— Page {source.get('page', '?')}"
                        )

                    # Save response in chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except requests.RequestException:
                    st.error(
                        "Cannot connect to FastAPI backend."
                    )

                except (KeyError, ValueError, TypeError):
                    st.error(
                        "Invalid response from backend."
                    )


# ---------- COMPARISON TAB ----------
with tab2:
    st.header("Compare Research Papers")

    if uploaded_files and len(uploaded_files) >= 2:
        names = [file.name for file in uploaded_files]

        col1, col2 = st.columns(2)

        with col1:
            paper1 = st.selectbox(
                "Select Paper 1", names
            )

        with col2:
            paper2 = st.selectbox(
                "Select Paper 2", names,
                index=1
            )

        if st.button("Compare Papers"):
            if paper1 == paper2:
                st.warning("Choose two different papers.")
            else:
                st.info(
                    f"Selected: {paper1} and {paper2}. "
                    "Comparison backend coming next."
                )
    else:
        st.info("Upload at least two PDFs to compare.")


# ---------- VISUAL TAB ----------
with tab3:
    st.header("Visual Research Analysis")

    image = st.file_uploader(
        "Upload a research diagram or chart",
        type=["png", "jpg", "jpeg"]
    )

    if image:
        st.image(image, caption="Selected research figure")
        st.info("AI figure interpretation coming next.")
