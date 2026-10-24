## PDG RAG Chatbot

Upload a PDF, ask questions, get answers with page-level citations.
FastAPI + LangChain + Qdrant + Gemini.

## Setup

```bash
git clone <your-repo> && cd pdf-rag-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your GOOGLE_API_KEY
docker compose up -d            # starts Qdrant on :6333
uvicorn app.main:app --reload
```

## Usage

```bash
# 1. Upload
curl -F "file=@paper.pdf" http://localhost:8000/upload
# {"doc_id":"a1b2c3d4","filename":"paper.pdf","pages":12,"chunks":58}

# 2. Ask
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"doc_id":"a1b2c3d4","question":"What problem does the paper solve?"}'
# {"answer":"The paper addresses ... [1][3]",
#  "sources":[{"ref":1,"page":2,"score":0.81,"snippet":"..."}, ...]}

# checking out all the docs
curl http://localhost:8000/documents
#deleting a document
curl -X DELETE http://localhost:8000/documents/a1b2c3d4
```
