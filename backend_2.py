import os
import socket
from urllib.parse import urlparse
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import io
import json
import httpx
from dotenv import load_dotenv

load_dotenv("backend.env")

app = FastAPI(title="PDF ReAct Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory store (Use Redis/S3 in production)
pdf_store = {}

# --- PDF Tool for the Agent ---
def get_pdf_context(file_id: str | None) -> str:
    """
    Return the full PDF text. 
    Grok-2 supports up to 128k/2M tokens, so passing the whole text is 
    much more accurate than naive substring searching.
    """
    if not file_id or file_id not in pdf_store:
        return ""
    return pdf_store[file_id]

# --- Groq LLM Setup ---
# Use `GROQ_API_KEY` and `GROQ_API_URL` in backend.env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.ai/v1/complete")

# --- API Endpoints ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str
    file_id: str | None = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
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
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in backend.env")

    # --- DEBUGGING ---
    print("\n" + "="*50)
    print(f"[DEBUG] API URL being used: {GROQ_API_URL}")
    print(f"[DEBUG] API Key prefix: {GROQ_API_KEY[:15]}...")
    print("="*50 + "\n")
    
        # DNS preflight to surface resolution errors clearly
    try:
        host = urlparse(GROQ_API_URL).netloc
        if host:
            socket.gethostbyname(host)
    except Exception as e:
        print(f"[ERROR] DNS resolution failed for host={host}: {e}")
        raise HTTPException(status_code=502, detail=(f"DNS resolution failed for Groq host '{host}': {e}. "
                                                         "Check GROQ_API_URL in backend.env and network/DNS settings."))

    context = get_pdf_context(request.file_id)
    
    # Safety truncation for massive PDFs
    if len(context) > 400000:
        context = context[:400000] + "\n\n[TRUNCATED DUE TO LENGTH]"
        
    system_prompt = (
        "You are a helpful AI assistant that analyzes PDF documents. "
        "Use the provided context when answering and cite any text you reference."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Context from PDF:\n{context}"})
    messages.append({"role": "user", "content": request.message})

    # Build a Groq-compatible payload (flexible/common fields)
    # Groq uses the OpenAI-compatible chat completions schema
    payload = {
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": messages,  # the list you already built above (system + context + user)
        "max_completion_tokens": int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "800")),
        "temperature": float(os.getenv("GROQ_TEMPERATURE", "0"))
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(GROQ_API_URL, json=payload, headers=headers)

            print(f"[DEBUG] Groq Status Code: {resp.status_code}")
            print(f"[DEBUG] Groq Response Body: {resp.text[:500]}")

            resp.raise_for_status()
            
        except httpx.HTTPStatusError as e:
            # FIX 2: Removed the emoji to prevent Windows UnicodeEncodeError
            print(f"[ERROR] GROQ API ERROR: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"Groq API error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            print(f"[ERROR] NETWORK ERROR: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")

    data = resp.json()

    # Flexible parsing for Groq response shapes
    data = resp.json()

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = json.dumps(data)[:2000]

    return {"reply": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)