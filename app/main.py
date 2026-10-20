import os
import tempfile

from fastapi import FastAPI,HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from app import rag

MAX_UPLOAD_MB= int(os.getenv("MAX_UPLOAD_MB","25"))
app= FastAPI(title="PDF RAG Chatbot", version="0.1.0")

class AskRequest(BaseModel):
    doc_id:str
    question: str= Field(min_length=1)
    k: int =Field(default=4, ge=1,le=12)

class Source(BaseModel):
    ref: int
    page: int
    score:float
    snippet:str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

@app.post("/upload")
def upload(file:UploadFile =File(...)):
    if not(file.filename or"").lower().endswith(".pdf"):
        raise HTTPException(400,"only pdf files are supported")
    data =file.file.read()
    if len(data)> MAX_UPLOAD_MB *1024*1024: 
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB.")

    tmp_path= None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path= tmp.name
        return rag.ingest_pdf(tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    
@app.post("/ask", response_model=AskResponse)
def ask(req:AskRequest):
    try:
        return rag.ask(req.doc_id, req.question, req.k)
    except KeyError:
        raise HTTPException(404, f"Unknown doc id: {req.doc_id}")

@app.get("/documents")
def documents():
    return rag.list_documents()

@app.delete("/documents/{doc_id}", status_code=204)
def delete(doc_id: str):
    try :
        rag.delete_docs(doc_id)
    except KeyError:
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")

@app.get("/health")
def health():
    return {"status": "ok"}