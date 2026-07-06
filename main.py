from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_core import init_rag, answer_question, ingest_texts


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_rag()
    yield


app = FastAPI(
    title="RAG API",
    description="API para consultar e alimentar a base RAG (Neo4j + Gemini)",
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pergunta em linguagem natural")


class AnswerResponse(BaseModel):
    question: str
    answer: str


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Lista de textos a inserir no índice")


class IngestResponse(BaseModel):
    inserted: int
    skipped: int
    total: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: QuestionRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question não pode ser vazia")

    try:
        result = answer_question(question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {e}")

    return AnswerResponse(question=question, answer=result)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
    try:
        result = ingest_texts(payload.texts)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao inserir documentos: {e}")

    return IngestResponse(**result)
