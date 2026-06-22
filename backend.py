import os
import json
import uuid
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import io
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
load_dotenv("backend.env") # This reads the .env file and sets the environment variables

# --- App Initialization ---
app = FastAPI(title="PDF ReAct Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory store for extracted PDF text (Use a DB like Redis/S3 in production)
pdf_store = {}

# This saves the conversation history in RAM while the server is running
checkpointer = MemorySaver()

# --- PDF Tool for the Agent ---
@tool
def search_pdf_content(query: str, config: RunnableConfig) -> str:
    """Searches the content of the uploaded PDF for the given query and returns relevant context."""
    file_id = config.get("configurable", {}).get("file_id")
    if not file_id or file_id not in pdf_store:
        return "Error: No PDF is currently loaded. Please upload a PDF first."
    
    text = pdf_store[file_id]
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Simple context window retrieval
    idx = text_lower.find(query_lower)
    if idx != -1:
        start = max(0, idx - 300)
        end = min(len(text), idx + 300)
        return f"Found context:\n...{text[start:end]}..."
    
    # Fallback: simple keyword split search
    words = query_lower.split()
    for word in words:
        idx = text_lower.find(word)
        if idx != -1:
            start = max(0, idx - 300)
            end = min(len(text), idx + 300)
            return f"Found partial context for '{word}':\n...{text[start:end]}..."

    return "No relevant information found in the PDF for this query."

# --- LangGraph ReAct Agent Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
tools = [search_pdf_content]

# We pass the checkpointer to enable conversation memory
agent = create_react_agent(
    model=llm, 
    tools=tools, 
    checkpointer=checkpointer,
    prompt="You are a helpful AI assistant that analyzes PDF documents. Use the search_pdf_content tool to find information. Always cite the text you find."
)

# --- API Endpoints ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str
    file_id: str | None = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF Ingestion API: Extracts text from the uploaded PDF."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
        
    file_id = str(uuid.uuid4())
    pdf_store[file_id] = text
    return {"file_id": file_id, "filename": file.filename, "pages": len(reader.pages)}

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """Streams the ReAct agent's execution via Server-Sent Events (SSE)."""
    
    async def event_generator() -> AsyncGenerator[str, None]:
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "file_id": request.file_id
            }
        }
        input_messages = {"messages": [HumanMessage(content=request.message)]}

        # Wrap the entire stream in a try/except block to catch backend crashes
        try:
            async for event in agent.astream_events(input_messages, config=config, version="v2"):
                event_kind = event["event"]
                
                # Stream LLM text tokens
                if event_kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # FIX: Gemini sometimes returns a list of dicts instead of a string
                        if isinstance(content, list):
                            text_content = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
                            if text_content:
                                yield f"data: {json.dumps({'type': 'token', 'content': text_content})}\n\n"
                        elif isinstance(content, str):
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                            
                # Stream Tool Calls (Agent reasoning)
                elif event_kind == "on_tool_start":
                    tool_input = event['data'].get('input', {})
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name'], 'input': str(tool_input)})}\n\n"
                    
                # Stream Tool Results
                elif event_kind == "on_tool_end":
                    output = str(event['data'].get('output', ''))
                    display_output = output[:200] + "..." if len(output) > 200 else output
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name'], 'output': display_output})}\n\n"

        except Exception as e:
            # If the backend crashes, catch it and send it to the UI + Terminal
            error_msg = f"Backend Error: {str(e)}"
            print(f"\n🔴 AGENT CRASHED: {error_msg}\n") # This will show in your backend terminal
            yield f"data: {json.dumps({'type': 'token', 'content': f'\n\n⚠️ **{error_msg}**'})}\n\n"

        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)