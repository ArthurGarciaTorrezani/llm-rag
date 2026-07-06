import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector

load_dotenv()

# ✅ Configuração via variáveis de ambiente (nunca hardcoded)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

INDEX_NAME = "index_bi"
KEYWORD_INDEX_NAME = "keywords_bi"
NODE_LABEL = "Chunk"
TEXT_NODE_PROPERTY = "text"
EMBEDDING_NODE_PROPERTY = "embedding"
GOOGLE_API_KEY = ""

# Estado global do módulo, populado por init_rag()
model = None
neo4j_vector_index = None


class SlicedGeminiEmbeddings:
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts):
        embeddings = self.base_embeddings.embed_documents(texts)
        return [vector[:768] for vector in embeddings]

    def embed_query(self, text):
        embedding = self.base_embeddings.embed_query(text)
        return embedding[:768]


def _load_seed_documents():
    documents = []
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    if os.path.isdir(data_dir):
        txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

        if not txt_files:
            print(f" Nenhum arquivo .txt encontrado em {data_dir}. Usando dados de teste.")
        for filename in txt_files:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                sentences = content.split(".")
                for sentence in sentences:
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 10:
                        documents.append(clean_sentence)
    else:
        print(f" Pasta {data_dir} não encontrada. Usando dados de teste.")
    return documents


def add_document_if_not_exists(text_content: str):
    global neo4j_vector_index

    try:
        search_results = neo4j_vector_index.similarity_search(text_content, k=1)
        if search_results and text_content in search_results[0].page_content:
            return False  # já existia
        neo4j_vector_index.add_texts([text_content])
        return True
    except Exception:
        # Se a busca falhar (ex: índice recém criado e ainda vazio), insere direto
        neo4j_vector_index.add_texts([text_content])
        return True


def init_rag(seed_if_empty: bool = True):
    global model, neo4j_vector_index

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY não definida. Configure o arquivo .env.")
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD não definida. Configure o arquivo .env.")

    print("Inicializando modelos Gemini...")
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        max_retries=2,
    )

    base_gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    gemini_embeddings = SlicedGeminiEmbeddings(base_gemini_embeddings)

    print("Conectando ao Neo4j e inicializando o índice...")
    neo4j_vector_index = Neo4jVector.from_texts(
        texts=[],
        embedding=gemini_embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name=INDEX_NAME,
        keyword_index_name=KEYWORD_INDEX_NAME,
        node_label=NODE_LABEL,
        text_node_property=TEXT_NODE_PROPERTY,
        embedding_node_property=EMBEDDING_NODE_PROPERTY,
    )

    if seed_if_empty:
        seed_documents = _load_seed_documents()
        inserted = 0
        for doc_text in seed_documents:
            if add_document_if_not_exists(doc_text):
                inserted += 1
            time.sleep(1.0)
        print(f"Seed concluído: {inserted}/{len(seed_documents)} documentos novos inseridos.")

    print("RAG inicializado.")


def ingest_texts(texts: list[str]) -> dict:
    if neo4j_vector_index is None:
        raise RuntimeError("RAG não inicializado. Chame init_rag() primeiro.")

    inserted, skipped = 0, 0
    for text in texts:
        clean = text.strip()
        if len(clean) <= 10:
            skipped += 1
            continue
        if add_document_if_not_exists(clean):
            inserted += 1
        else:
            skipped += 1

    return {"inserted": inserted, "skipped": skipped, "total": len(texts)}


def answer_question(question: str) -> str:
    if neo4j_vector_index is None or model is None:
        raise RuntimeError("RAG não inicializado. Chame init_rag() primeiro.")

    results = neo4j_vector_index.similarity_search_with_score(question, k=1)

    relevant_chunks = []
    for doc, score in results:
        if doc.page_content:
            clean_text = doc.page_content.replace("text: ", "")
            relevant_chunks.append(clean_text)

    if not relevant_chunks:
        return "Sorry, I couldn't find enough information to answer."

    context = "\n".join(relevant_chunks)

    prompt = f"""
    Answer the question concisely and naturally based on the following context:
    Don't use information outside of the provided context.

    Context:
    {context}

    Question: {question}

    Provide a direct and informative response:
    """

    response = model.invoke(prompt)
    return response.content
