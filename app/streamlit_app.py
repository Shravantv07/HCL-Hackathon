
import os
from uuid import uuid4

import requests
import streamlit as st

# ---------------- SETTINGS ----------------
st.set_page_config(
    page_title="ResearchIQ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv(
    "RESEARCHIQ_API_URL",
    "http://127.0.0.1:8000"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: #101114;
        color: #f5f5f5;
    }

    [data-testid="stSidebar"] {
        background: #17181c;
        border-right: 1px solid #303136;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    .brand {
        font-size: 24px;
        font-weight: 750;
        color: #f5f5f5;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #989ba4;
        font-size: 13px;
    }

    .welcome {
        text-align: center;
        padding-top: 12vh;
        padding-bottom: 25px;
    }

    .welcome-title {
        font-size: 32px;
        font-weight: 650;
        color: #f2f2f2;
        letter-spacing: -1px;
    }

    .welcome-subtitle {
        color: #a1a1aa;
        font-size: 15px;
        margin-top: 12px;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding-top: 15px;
        padding-bottom: 15px;
    }

    [data-testid="stChatInput"] {
        border: 1px solid #42444a;
        border-radius: 20px;
        background: #23252b;
    }

    .stButton button {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- CHAT STATE ----------------
def create_chat():
    chat_id = uuid4().hex

    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": []
    }

    st.session_state.active_chat = chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}
    create_chat()

if "active_chat" not in st.session_state:
    create_chat()

active_id = st.session_state.active_chat


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand">🔬 ResearchIQ</div>',
        unsafe_allow_html=True
    )
    st.caption("AI Research Companion")

    st.divider()

    if st.button(
        "＋  New Chat",
        use_container_width=True
    ):
        create_chat()
        st.rerun()

    st.write("")

    st.caption("CHAT HISTORY")

    for chat_id, chat_data in reversed(
        list(st.session_state.chats.items())
    ):
        if st.button(
            "💬 " + chat_data["title"],
            key=f"history_{chat_id}",
            use_container_width=True
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

    st.divider()

    st.caption("RESEARCH TOPIC")
    st.write("Agentic AI & Multi-Agent Systems")
    st.caption("ResearchIQ • Hackathon Prototype")


# ---------------- MAIN INTERFACE ----------------
chat = st.session_state.chats[active_id]

st.markdown("### 🔬 ResearchIQ")
st.caption("Multimodal AI Research Assistant")
st.divider()

if not chat["messages"]:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">
            What are we researching today?
        </div>
        <div class="welcome-subtitle">
            Ask questions, upload research papers,
            and explore scientific insights.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- DISPLAY CHAT HISTORY ----------------
for message in chat["messages"]:
    with st.chat_message(message["role"]):
        if message.get("content"):
            st.markdown(message["content"])

        for attachment in message.get("attachments", []):
            if attachment["type"] == "image":
                st.image(
                    attachment["data"],
                    caption=attachment["name"],
                    width=280
                )
            else:
                st.caption("📄 " + attachment["name"])

        for source in message.get("sources", []):
            st.caption(
                f"📄 {source.get('paper', 'Unknown')} "
                f"— Page {source.get('page', '?')}"
            )


# ---------------- CHAT COMPOSER ----------------
submission = st.chat_input(
    "Ask ResearchIQ anything or attach papers...",
    accept_file="multiple",
    file_type=["pdf", "png", "jpg", "jpeg"],
    key=f"composer_{active_id}"
)

if submission:
    question = submission.text.strip()
    files = submission.files

    if question or files:
        attachments = []

        # Store files for display in chat
        for file in files:
            is_pdf = file.name.lower().endswith(".pdf")

            attachment = {
                "name": file.name,
                "type": "pdf" if is_pdf else "image"
            }

            if not is_pdf:
                attachment["data"] = file.getvalue()

            attachments.append(attachment)

        chat["messages"].append({
            "role": "user",
            "content": question,
            "attachments": attachments
        })

        # Name chat based on first question
        if chat["title"] == "New Chat":
            chat["title"] = (
                question[:35] if question
                else "Uploaded research files"
            )

        response_parts = []
        sources = []

        with st.spinner("Working on your request..."):

            # Send PDF attachments to FastAPI
            for file in files:
                if not file.name.lower().endswith(".pdf"):
                    continue

                try:
                    upload_response = requests.post(
                        f"{API_URL}/upload",
                        files={
                            "file": (
                                file.name,
                                file.getvalue(),
                                "application/pdf"
                            )
                        },
                        timeout=60
                    )

                    upload_response.raise_for_status()

                    response_parts.append(
                        f"✅ Uploaded **{file.name}** "
                        "(not indexed yet)."
                    )

                except requests.RequestException:
                    response_parts.append(
                        f"❌ Failed to upload **{file.name}**."
                    )

            
            # Send image attachments to FastAPI
            for file in files:
                if file.name.lower().endswith(".pdf"):
                    continue

                with st.spinner(f"Uploading {file.name}..."):
                    try:
                        extension = file.name.lower().split(".")[-1]
                        mime_type = (
                            "image/png" if extension == "png"
                            else "image/jpeg"
                        )

                        response = requests.post(
                            f"{API_URL}/upload-image",
                            files={
                                "file": (
                                    file.name,
                                    file.getvalue(),
                                    mime_type
                                )
                            },
                            timeout=60
                        )

                        response.raise_for_status()
                        result = response.json()

                        response_parts.append(
                            f"✅ Image **{file.name}** uploaded successfully. "
                            f"ID: `{result['image_id']}`"
                        )

                    except requests.RequestException:
                        response_parts.append(
                            f"❌ Failed to upload image **{file.name}**."
                        )


            # Send question to FastAPI
            if question:
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question,
                            "selected_papers": []
                        },
                        timeout=60
                    )

                    response.raise_for_status()
                    result = response.json()

                    response_parts.append(
                        result.get(
                            "answer",
                            "No answer returned."
                        )
                    )

                    sources = result.get("sources", [])

                except requests.RequestException:
                    response_parts.append(
                        "⚠️ Unable to connect to "
                        "the FastAPI backend."
                    )

                except (ValueError, TypeError):
                    response_parts.append(
                        "⚠️ Invalid response from the backend."
                    )

        chat["messages"].append({
            "role": "assistant",
            "content": "\n\n".join(response_parts),
            "sources": sources
        })

        st.rerun()
