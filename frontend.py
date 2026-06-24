import streamlit as st
import httpx
import json
import uuid

# --- Page Config ---
st.set_page_config(page_title="PDF ReAct Agent", layout="wide")
st.title("📄 PDF ReAct Agent with LangGraph")

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

BACKEND_URL = "http://localhost:8000"

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and uploaded_file.name != st.session_state.get("uploaded_filename"):
        with st.spinner("Extracting text from PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = httpx.post(f"{BACKEND_URL}/upload", files=files, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                st.session_state.file_id = data["file_id"]
                st.session_state.uploaded_filename = uploaded_file.name
                st.success(f"Loaded: {data['filename']} ({data['pages']} pages)")
            else:
                st.error("Failed to upload PDF.")

    st.header("2. Thread Management")
    new_thread = st.button("➕ New Thread")

    if new_thread:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.file_id = None
        st.rerun()

    st.subheader("Or load existing thread")

    input_thread = st.text_input("Enter Thread ID")

    # Load Thread button
    if st.button("Load Thread"):
        try:
            resp = httpx.get(f"{BACKEND_URL}/thread/{input_thread}", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.thread_id = input_thread
            st.session_state.file_id = data.get("file_id")
            st.session_state.messages = data.get("messages", [])
        except Exception as e:
            st.error(f"Could not load thread: {e}")
        st.rerun()

    if st.button("Clear Chat & Memory"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.file_id = None
        st.session_state.uploaded_filename = None
        st.rerun()

    st.divider()
    st.caption(f"Thread ID: {st.session_state.thread_id}")
    st.caption(f"**File ID:** `{st.session_state.file_id[:8] if st.session_state.file_id else 'None'}...`")

# --- Helper: Call Chat Endpoint ---
def get_chat_response(message: str):
    url = f"{BACKEND_URL}/chat"
    payload = {
        "message": message,
        "thread_id": st.session_state.thread_id
    }
    response = httpx.post(
        f"{BACKEND_URL}/upload",
        files=files,
        data={"thread_id": st.session_state.thread_id},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["reply"]

# --- Main Chat Interface ---
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "thoughts" in msg:
            with st.expander("View Agent Reasoning & Tool Calls"):
                for thought in msg["thoughts"]:
                    st.json(thought)

# Chat input
if prompt := st.chat_input("Ask a question about the PDF..."):
    if not st.session_state.file_id:
        st.warning("Please upload a PDF in the sidebar first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    full_response = get_chat_response(prompt)
                except httpx.HTTPStatusError as e:
                    full_response = f"Error: {e.response.status_code} - {e.response.text}"
                except Exception as e:
                    full_response = f"Error: {str(e)}"

            st.markdown(full_response)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })