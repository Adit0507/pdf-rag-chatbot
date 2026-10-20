import os
import re
import uuid
from  functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL= os.getenv("QDRANT_URL", "http://localhost:6333")
MODEL = os.getenv("GEMINI_MODEL","gemini-3.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
PREFIX = "pdf_"  

PROMPT =ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a PDF using ONLY the numbered context "
            "excerpts provided. Cite every claim with the excerpt number in square "
            "brackets, e.g. [1] or [2][3]. If the excerpts don't contain the answer, "
            "reply exactly: \"I couldn't find that in the document.\" and cite nothing. "
            "Don't use outside knowledge.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

@lru_cache(maxsize=1)
def _embeddings()-> FastEmbedEmbeddings:
    return  FastEmbedEmbeddings(model_name=EMBED_MODEL)

@lru_cache(maxsize=1)
def _client()-> QdrantClient:
    return QdrantClient(url=QDRANT_URL)

@lru_cache(maxsize=1)
def _chain():
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, max_output_tokens=2048,)
    return PROMPT | llm | StrOutputParser()

def _collection(doc_id: str)-> str:
    return f"{PREFIX}{doc_id}"

def ingest_pdf(path:str, filename: str)-> dict:
    pages= PyPDFLoader(path).load()
    for p in pages:
        p.page_content =re.sub(r"\s+", " ", p.page_content).strip()

    if not any(p.page_content.strip() for p in pages):
        raise ValueError("No extractable text found (scanned PDF? OCR isn't supported).")

    splitter =RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks =splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata= {
            "source": filename,
            "page": int(c.metadata.get("page", 0)) + 1, 
            "chunk": i,
        }

    doc_id=uuid.uuid4().hex[:8]
    QdrantVectorStore.from_documents(chunks, _embeddings(),url=QDRANT_URL, collection_name=_collection(doc_id))
    return {"doc_id": doc_id, "filename": filename, "pages": len(pages), "chunks": len(chunks)}

def list_documents()-> list[dict]:
    client =_client()
    docs =[]

    for col in client.get_collections().collections:
        if not col.name.startswith(PREFIX):
            continue

        points, _=client.scroll(col.name, limit=1,with_payload=True)
        meta =(points[0].payload or {}).get("metadata", {}) if points else {}
        docs.append({
            "doc_id": col.name[len(PREFIX):],
            "filename": meta.get("source", "unknown"),
            "chunks": client.count(col.name).count,
        })

    return docs

def delete_docs(doc_id:str)-> None:
    client= _client()
    if not client.collection_exists(_collection(doc_id)):
        raise KeyError(doc_id)
    client.delete_collection(_collection(doc_id))

def ask(doc_id: str, question:str, k:int= 4) ->dict:
    client =_client()
    name= _collection(doc_id)
    if not client.collection_exists(name):
        raise KeyError(doc_id)

    store=QdrantVectorStore(client=client,collection_name=name,embedding=_embeddings())
    hits =store.similarity_search_with_score(question,k =k)

    context = "\n\n".join(
        f"[{i}] (page {doc.metadata['page']})\n{doc.page_content}"
        for i, (doc, _) in enumerate(hits, start=1)
    )
    answer =_chain().invoke({"context":context, "question": question})
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    sources = [
        {
            "ref": i,
            "page": hits[i - 1][0].metadata["page"],
            "score": round(float(hits[i - 1][1]), 4),
            "snippet": hits[i - 1][0].page_content[:300].strip(),
        }
        for i in cited
        if 1 <= i <= len(hits)
    ]
    return {"answer": answer, "sources": sources} 