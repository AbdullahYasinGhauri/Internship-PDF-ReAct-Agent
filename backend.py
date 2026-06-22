import os
import json
import uuid
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import json
import io
import httpx
import socket
import ssl
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv("backend.env") # This reads the .env file and sets the environment variables

# --- App Initialization ---
app = FastAPI(title="PDF ReAct Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory store for extracted PDF text (Use a DB like Redis/S3 in production)
pdf_store = {}

# --- PDF Tool for the Agent ---
def get_pdf_context(file_id: str | None, query: str) -> str:
    """Return a short context snippet from the uploaded PDF for the given query."""
    if not file_id or file_id not in pdf_store:
        return ""

    text = pdf_store[file_id]
    query_lower = (query or "").lower()
    text_lower = text.lower()

    if not query_lower:
        return text[:1000]

    # Simple context window retrieval
    idx = text_lower.find(query_lower)
    if idx != -1:
        start = max(0, idx - 300)
        end = min(len(text), idx + 300)
        return text[start:end]

    # Fallback: simple keyword split search
    for word in query_lower.split():
        idx = text_lower.find(word)
        if idx != -1:
            start = max(0, idx - 300)
            end = min(len(text), idx + 300)
            return text[start:end]

    return ""

# --- Grok (x.com) LLM Setup ---
# We call the Grok HTTP API directly (non-streaming). Configure via backend.env
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = os.getenv("GROK_API_URL", "https://api.grok.x.com/v1/generate")

if not GROK_API_KEY:
    # It's okay to run locally without a key for some endpoints; we raise at request time instead.
    pass

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
async def chat(request: ChatRequest):
    """Handle chat requests using Grok (non-streaming)."""
    if not GROK_API_KEY:
        raise HTTPException(status_code=500, detail="GROK_API_KEY not set in backend.env")

    # Build prompt with optional PDF context
    context = get_pdf_context(request.file_id, request.message)
    system_prompt = (
        "You are a helpful AI assistant that analyzes PDF documents. "
        "Use the provided context when answering and cite any text you reference."
    )

    full_prompt = system_prompt + "\n\n"
    if context:
        full_prompt += "Context from PDF:\n" + context + "\n\n"
    full_prompt += "User: " + request.message

    # Call Grok API (best-effort generic REST payload)
    # Quick DNS check to provide a clearer error than getaddrinfo failing within httpx
    try:
        host = urlparse(GROK_API_URL).netloc
        if host:
            socket.gethostbyname(host)
    except Exception as e:
        print(f"[Grok] DNS resolution failed for host={host}: {e}")
        raise HTTPException(status_code=502, detail=(f"DNS resolution failed for Grok host '{host}': {e}. "
                                                     "Check GROK_API_URL in backend.env and network/DNS settings."))

    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": "grok-2-latest",  # You must specify the model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Context from PDF:\n{context}"},
                {"role": "user", "content": request.message}
            ],
            "max_tokens": 800,
            "temperature": 0
}

        # Primary attempt: Authorization: Bearer
        headers_bearer = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

        try:
            resp = await client.post(GROK_API_URL, json=payload, headers=headers_bearer)
        except Exception as e:
            # Network-level error; handle SSL SNI/unrecognized name by retrying insecurely once
            err_str = str(e)
            print(f"[Grok] network error: {err_str}")
            if isinstance(e, ssl.SSLError) or "unrecognized name" in err_str.lower():
                # Retry with verification disabled to work around TLS/SNI middlebox issues
                print("[Grok] SSL error detected; retrying request with verify=False (insecure)")
                try:
                    async with httpx.AsyncClient(timeout=60.0, verify=False) as insecure_client:
                        resp = await insecure_client.post(GROK_API_URL, json=payload, headers=headers_bearer)
                except Exception as e2:
                    print(f"[Grok] insecure retry failed: {e2}")
                    raise HTTPException(status_code=502, detail=f"Error contacting Grok API: {e2}")
            else:
                raise HTTPException(status_code=502, detail=f"Error contacting Grok API: {e}")

        # If we get an auth error or other 4xx/5xx, try a fallback header (`x-api-key`) once
        if resp.status_code >= 400:
            resp_text_snip = resp.text[:2000]
            print(f"[Grok] first attempt status={resp.status_code} text={resp_text_snip}")

            # Try fallback header
            headers_x = {"x-api-key": GROK_API_KEY, "Content-Type": "application/json"}
            try:
                resp2 = await client.post(GROK_API_URL, json=payload, headers=headers_x)
            except Exception as e:
                print(f"[Grok] fallback network error: {e}")
                raise HTTPException(status_code=502, detail=f"Error contacting Grok API (fallback): {e}")

            if resp2.status_code < 400:
                resp = resp2
            else:
                # Still failing: print both responses and return a helpful error
                resp2_text_snip = resp2.text[:2000]
                print(f"[Grok] fallback attempt status={resp2.status_code} text={resp2_text_snip}")
                raise HTTPException(status_code=502, detail=(f"Grok API error: first status={resp.status_code} "
                                                            f"text={resp_text_snip} | fallback status={resp2.status_code} "
                                                            f"text={resp2_text_snip}"))

    # Flexible parsing of possible response formats
    try:
        data = resp.json()
    except Exception:
        text = resp.text
    else:
        text = ""
        # Common patterns
        if isinstance(data, dict):
            if "output" in data:
                text = data["output"]
            elif "text" in data:
                text = data["text"]
            elif "choices" in data and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict):
                    text = first.get("text") or first.get("message", {}).get("content", "")
        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                text = first.get("text", "")

    text = text or resp.text

    return {"reply": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)