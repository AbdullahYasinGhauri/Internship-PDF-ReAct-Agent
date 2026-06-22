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
    
    if uploaded_file and st.session_state.file_id is None:
        with st.spinner("Extracting text from PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = httpx.post(f"{BACKEND_URL}/upload", files=files, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                st.session_state.file_id = data["file_id"]
                st.success(f"Loaded: {data['filename']} ({data['pages']} pages)")
            else:
                st.error("Failed to upload PDF.")

    if st.button("Clear Chat & Memory"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
    st.caption(f"**File ID:** `{st.session_state.file_id[:8] if st.session_state.file_id else 'None'}...`")

# --- Helper: Consume SSE Stream ---
def stream_chat(message: str):
    url = f"{BACKEND_URL}/chat"
    payload = {
        "message": message,
        "thread_id": st.session_state.thread_id,
        "file_id": st.session_state.file_id
    }
    
    # We use httpx to stream the SSE response
    with httpx.stream("POST", url, json=payload, timeout=None) as response:
        buffer = ""
        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                sse_message, buffer = buffer.split("\n\n", 1)
                for line in sse_message.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            return
                        
                        data = json.loads(data_str)
                        yield data

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

        # Stream assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            thoughts_placeholder = st.empty()
            
            full_response = ""
            thoughts = []
            current_thought = ""
            
            with thoughts_placeholder.container():
                thought_status = st.status("Agent is thinking...", expanded=True)

            # Process SSE Stream
            for data in stream_chat(prompt):
                if data["type"] == "token":
                    full_response += data["content"]
                    message_placeholder.markdown(full_response + "▌")
                    
                elif data["type"] == "tool_start":
                    current_thought = f"🛠️ Calling Tool: **{data['tool']}**\n\nInput: `{data['input']}`"
                    with thought_status:
                        st.markdown(current_thought)
                        
                elif data["type"] == "tool_end":
                    current_thought += f"\n\n✅ Tool Output:\n```\n{data['output']}\n```"
                    with thought_status:
                        st.markdown(current_thought)
            
            # Finalize UI
            thought_status.update(label="Agent finished reasoning", state="complete", expanded=False)
            message_placeholder.markdown(full_response)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "thoughts": thoughts # (Optional: save thoughts to history if needed)
            })