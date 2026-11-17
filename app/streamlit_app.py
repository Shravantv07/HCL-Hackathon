
import os
from uuid import uuid4
import time
import requests
import streamlit as st

# ---------------- SETTINGS ----------------
st.set_page_config(
    page_title="ResearchIQ | VECTOR",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CINEMATIC STARTUP ANIMATION ----------

if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

if not st.session_state.intro_shown:

    splash = st.empty()

    splash.markdown("""
    <style>
    .vector-intro {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        background: #000000;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        animation: introExit 2.8s ease-in-out forwards;
    }

    .vector-title {
        font-family: Arial, sans-serif;
        font-size: clamp(36px, 7vw, 74px);
        font-weight: 300;
        letter-spacing: 14px;
        color: #ffffff;
        opacity: 0;
        animation: textReveal 2s ease forwards;
    }

    .vector-subtitle {
        font-family: Arial, sans-serif;
        font-size: 13px;
        letter-spacing: 9px;
        color: #999999;
        margin-top: 18px;
        opacity: 0;
        animation: textReveal 1.5s ease 0.6s forwards;
    }

    @keyframes textReveal {
        0% {
            opacity: 0;
            transform: translateY(15px);
            filter: blur(8px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
        }
    }

    @keyframes introExit {
        0%, 70% {
            opacity: 1;
        }
        100% {
            opacity: 0;
            visibility: hidden;
        }
    }
    </style>

    <div class="vector-intro">
        <div class="vector-title">VECTOR</div>
        <div class="vector-subtitle">PRESENTS</div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(2.8)

    splash.empty()
    st.session_state.intro_shown = True

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


@st.cache_data(ttl=30, show_spinner=False)
def fetch_papers():
    """Indexed papers (team catalogue + uploads) from the backend; None if unreachable."""
    try:
        response = requests.get(f"{API_URL}/papers", timeout=10)
        response.raise_for_status()
        return [p for p in response.json().get("papers", []) if p.get("status") == "indexed"]
    except (requests.RequestException, ValueError):
        return None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand">⚡ ResearchIQ</div>',
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

    # ---- Paper library: limit answers to chosen papers, manage uploads ----
    st.caption("PAPER LIBRARY")
    library = fetch_papers()
    if library is None:
        st.caption("⚠️ Backend not reachable.")
    else:
        labels = {p["paper_id"]: (p.get("title") or p["paper_id"])[:70] for p in library}
        # Drop selections that no longer exist (e.g. a deleted upload).
        st.session_state["paper_filter"] = [
            pid for pid in st.session_state.get("paper_filter", []) if pid in labels
        ]
        st.multiselect(
            "Limit answers to",
            options=list(labels),
            format_func=lambda pid: labels[pid],
            placeholder=f"All {len(labels)} papers",
            key="paper_filter",
        )
        uploads = [p for p in library if p.get("source") == "upload"]
        if uploads:
            with st.expander(f"Your uploads ({len(uploads)})"):
                for p in uploads:
                    col_name, col_delete = st.columns([5, 1])
                    col_name.caption(labels[p["paper_id"]])
                    if col_delete.button("🗑", key=f"delete_{p['paper_id']}", help="Remove from library"):
                        try:
                            requests.delete(f"{API_URL}/papers/{p['paper_id']}", timeout=60).raise_for_status()
                        except requests.RequestException:
                            st.toast("Could not delete that upload.")
                        fetch_papers.clear()
                        st.rerun()

    st.toggle(
        "Verify claims (slower)",
        key="verify_claims",
        help="Checks every claim in the answer against its cited passage.",
    )

    st.divider()

    st.caption("RESEARCH TOPIC")
    st.write("Multimodal Large Language Models")
    st.caption("ResearchIQ • Hackathon Prototype")


# ---------------- MAIN INTERFACE ----------------
chat = st.session_state.chats[active_id]

st.markdown("### ⚡ ResearchIQ")
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
ROUTE_LABELS = {
    "comparison": "📊 Paper comparison",
    "synthesis": "📚 Literature synthesis",
    "visual": "🖼️ Figure analysis",
}
VERDICT_ICONS = {
    "supported": "✅", "partially_supported": "🟡", "unsupported": "❌",
    "contradicted": "⛔", "interpretation": "💭", "not_a_claim": "▫️", "unverified": "❔",
}


def render_extras(message):
    """Route badge, warnings, labelled sources with evidence, and the evidence check."""
    if message.get("route") in ROUTE_LABELS:
        st.caption(ROUTE_LABELS[message["route"]])

    for warning in (message.get("warnings") or [])[:4]:
        st.caption("⚠️ " + warning)

    sources = message.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                label = source.get("source_label")
                where = [f"p.{source['page']}" if source.get("page") else "page unknown"]
                if source.get("section"):
                    where.append(source["section"])
                if source.get("fig_ref"):
                    where.append(source["fig_ref"])
                st.markdown(
                    (f"**[{label}]** " if label else "")
                    + f"{source.get('paper', 'Unknown')} — " + " · ".join(where)
                )
                text = (source.get("text") or "").strip()
                if text:
                    st.caption(text[:700] + ("…" if len(text) > 700 else ""))

    verification = message.get("verification")
    if verification and verification.get("claims"):
        summary = ", ".join(
            f"{count} {verdict.replace('_', ' ')}"
            for verdict, count in verification.get("summary", {}).items()
        )
        title = {
            "verified": "✅ Evidence check: all claims supported",
            "issues_found": "🟡 Evidence check: some claims need attention",
        }.get(verification.get("status"), "Evidence check")
        with st.expander(f"{title} ({summary})"):
            for claim in verification["claims"]:
                icon = VERDICT_ICONS.get(claim["verdict"], "•")
                st.markdown(f"{icon} **{claim['verdict'].replace('_', ' ')}** — {claim['text']}")
                if claim.get("issues"):
                    st.caption("; ".join(claim["issues"]))
    elif verification and verification.get("status") == "error":
        st.caption(f"⚠️ Evidence check failed: {verification.get('error')}")


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

        if message["role"] == "assistant":
            render_extras(message)


# ---------------- CHAT COMPOSER ----------------
DOC_EXTENSIONS = (".pdf", ".txt", ".md")

submission = st.chat_input(
    "Ask ResearchIQ anything or attach papers...",
    accept_file="multiple",
    file_type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
    key=f"composer_{active_id}"
)

if submission:
    question = submission.text.strip()
    files = submission.files

    if question or files:
        attachments = []

        # Store files for display in chat
        for file in files:
            is_doc = file.name.lower().endswith(DOC_EXTENSIONS)

            attachment = {
                "name": file.name,
                "type": "pdf" if is_doc else "image"
            }

            if not is_doc:
                attachment["data"] = file.getvalue()

            attachments.append(attachment)

        # Recent turns let the backend resolve follow-ups ("what about its limitations?")
        history = [
            {"role": m["role"], "content": (m.get("content") or "")[:1500]}
            for m in chat["messages"][-4:]
            if m.get("content")
        ]

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
        result = {}
        uploaded_pdf_ids = []
        failed_pdfs = []
        image_ids = []

        # Send document attachments to FastAPI (indexing happens during upload)
        for file in files:
            if not file.name.lower().endswith(DOC_EXTENSIONS):
                continue

            with st.spinner(f"Reading and indexing {file.name}… large papers can take a minute."):
                try:
                    upload_response = requests.post(
                        f"{API_URL}/upload",
                        files={
                            "file": (
                                file.name,
                                file.getvalue(),
                                "application/pdf" if file.name.lower().endswith(".pdf")
                                else "text/plain"
                            )
                        },
                        timeout=900  # upload includes parsing, OCR, figure descriptions, embeddings
                    )

                    upload_response.raise_for_status()
                    upload_result = upload_response.json()
                    uploaded_pdf_ids.append(upload_result["paper_id"])
                    fetch_papers.clear()  # refresh the sidebar library
                    title = upload_result.get("title") or file.name
                    if upload_result.get("duplicate"):
                        response_parts.append(
                            f"✅ **{title}** is already in the library."
                        )
                    else:
                        response_parts.append(
                            f"✅ Indexed **{title}** — "
                            f"{upload_result.get('pages')} pages, "
                            f"{upload_result.get('text_chunks')} text chunks, "
                            f"{upload_result.get('figures')} figures."
                        )

                except requests.HTTPError as e:
                    failed_pdfs.append(file.name)
                    try:
                        detail = e.response.json().get("detail", "")
                    except ValueError:
                        detail = ""
                    response_parts.append(
                        f"❌ Could not add **{file.name}**"
                        + (f": {detail}" if detail else ".")
                    )

                except requests.RequestException:
                    failed_pdfs.append(file.name)
                    response_parts.append(
                        f"❌ Failed to upload **{file.name}**."
                    )

        # Send image attachments to FastAPI
        for file in files:
            if file.name.lower().endswith(DOC_EXTENSIONS):
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
                    image_ids.append(response.json()["image_id"])
                    if not question:
                        response_parts.append(
                            f"✅ Image **{file.name}** received. Ask a question about it."
                        )

                except requests.RequestException:
                    response_parts.append(
                        f"❌ Failed to upload image **{file.name}**."
                    )

        # A question about attached PDFs must not fall back to the whole library
        # when those PDFs could not be read: the answer would be about other papers.
        if question and failed_pdfs and not uploaded_pdf_ids and not image_ids:
            response_parts.append(
                "I didn't answer your question because the attached file could not be "
                "read, and answering from the rest of the library would not be about "
                "your file. Try a PDF with selectable text, or ask without an attachment "
                "to search the whole library."
            )
            question = ""
        elif question and failed_pdfs:
            response_parts.append(
                f"Answering from your other attachment(s) only; "
                f"{', '.join(failed_pdfs)} could not be read."
            )

        # Send question to FastAPI (it picks QA, comparison, synthesis or figure analysis)
        if question:
            with st.spinner("Searching the papers and writing an answer…"):
                try:
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question,
                            "selected_papers": st.session_state.get("paper_filter", []),
                            "uploaded_pdf_ids": uploaded_pdf_ids,
                            "image_ids": image_ids,
                            "history": history,
                            "verify": st.session_state.get("verify_claims", False),
                        },
                        timeout=900  # synthesis across many papers can take a few minutes
                    )

                    response.raise_for_status()
                    result = response.json()

                    response_parts.append(
                        result.get(
                            "answer",
                            "No answer returned."
                        )
                    )

                except requests.HTTPError as e:
                    try:
                        detail = e.response.json().get("detail", "")
                    except ValueError:
                        detail = ""
                    response_parts.append(f"⚠️ The backend returned an error. {detail}".strip())

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
            "sources": result.get("sources", []),
            "route": result.get("route"),
            "warnings": result.get("warnings", []),
            "verification": result.get("verification"),
        })

        st.rerun()
