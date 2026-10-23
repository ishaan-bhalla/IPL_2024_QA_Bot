import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = os.path.join(ROOT, "data", "ipl_2024_wikipedia.txt")  # refresh with scripts/snapshot_wikipedia.py
TOP_K = 4  # chunks retrieved per question
CANDIDATES = 20  # chunks each retriever proposes before fusion
GROQ_URL = "https://api.groq.com/openai/v1"
WIKI_URL = "https://en.wikipedia.org/wiki/2024_Indian_Premier_League"


def load_sections() -> tuple[str, list[tuple[str, str]]]:
    """Return the snapshot's source URL and its (section, text) pairs."""
    with open(SNAPSHOT) as f:
        header, _, body = f.read().partition("\n\n")
    source = header.splitlines()[0].removeprefix("SOURCE: ").strip()
    sections = []
    for i, block in enumerate(body.split("\n## ")):
        title, _, text = block.partition("\n") if i else ("Summary", "", block)
        if text.strip():
            sections.append((title.strip(), text.strip()))
    return source, sections


@lru_cache(maxsize=1)
def get_vector_store() -> tuple[FAISS, BM25Retriever]:
    """Split each article section into chunks and index them once per process, for embedding
    search and keyword (BM25) search. Each chunk starts with its section name, so a question
    like "who won the Emerging Player award" can match the awards table by its heading."""
    source, sections = load_sections()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for title, text in sections:
        for piece in splitter.split_text(text):
            chunks.append(Document(page_content=f"{title}: {piece}",
                                   metadata={"section": title, "source": source, "chunk": len(chunks)}))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings), BM25Retriever.from_documents(chunks, k=CANDIDATES)


def retrieve(question: str, k: int = TOP_K) -> list[Document]:
    """Hybrid search: merge the embedding and BM25 rankings with reciprocal rank fusion.
    Embeddings alone kept returning the long Fixtures table for table questions."""
    vector_store, bm25 = get_vector_store()
    scores, docs = {}, {}
    for ranking in (vector_store.similarity_search(question, k=CANDIDATES), bm25.invoke(question)):
        for rank, doc in enumerate(ranking):
            i = doc.metadata["chunk"]
            scores[i] = scores.get(i, 0) + 1 / (60 + rank)
            docs[i] = doc
    return [docs[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]


def pick_groq_models(api_key: str) -> list[str]:
    """Chat models Groq currently serves, best first (it renames and retires models)."""
    import httpx
    ids = [m["id"] for m in httpx.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {api_key}"},
                                      timeout=20).raise_for_status().json()["data"]]
    chat = [i for i in ids if not any(x in i.lower() for x in ("guard", "whisper", "orpheus", "tts"))]
    if not chat:
        raise RuntimeError("No chat model available on Groq")
    ranked = []
    for pref in ("gpt-oss-120b", "llama", "gpt-oss-20b", "qwen"):
        ranked += [m for m in chat if pref in m.lower() and m not in ranked]
    return ranked + [m for m in chat if m not in ranked]


@lru_cache(maxsize=1)
def get_llm():
    """Groq (free tier) when GROQ_API_KEY is set, with a second Groq model as fallback;
    otherwise GPT-4o when OPENAI_API_KEY is set; otherwise a local Ollama model."""
    from langchain_openai import ChatOpenAI
    if os.getenv("GROQ_API_KEY"):
        key = os.getenv("GROQ_API_KEY")
        models = pick_groq_models(key)
        if os.getenv("GROQ_MODEL"):
            models = [os.getenv("GROQ_MODEL")] + [m for m in models if m != os.getenv("GROQ_MODEL")]

        def groq(model):
            return ChatOpenAI(base_url=GROQ_URL, api_key=key, model=model, temperature=0, timeout=60, max_retries=3)

        primary = groq(models[0])
        return primary.with_fallbacks([groq(models[1])]) if len(models) > 1 else primary
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o", temperature=0, timeout=60, max_retries=3)
    return ChatOpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1", api_key="ollama",
                      model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"), temperature=0, timeout=300)


def make_prompt(question: str, passages: list) -> str:
    context = "\n\n".join(doc.page_content for doc in passages)  # each chunk starts with its section name
    return (
        "Answer the question about the 2024 Indian Premier League using only the passages below, which come from "
        "the Wikipedia article. Table rows are written as 'cell | cell | cell'.\n"
        "- Start with the answer itself, in one to three sentences.\n"
        "- If the passages don't contain the answer, say you couldn't find it in the article instead of guessing.\n\n"
        f"PASSAGES:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )


def answer_with_stats(question: str) -> dict:
    """Answer a question and report the sections used, retrieval and LLM time, and token counts."""
    started = time.perf_counter()
    passages = retrieve(question)
    retrieval_ms = round((time.perf_counter() - started) * 1000)
    sections = list(dict.fromkeys(doc.metadata["section"] for doc in passages))

    started = time.perf_counter()
    reply = get_llm().invoke(make_prompt(question, passages))
    llm_ms = round((time.perf_counter() - started) * 1000)
    usage = getattr(reply, "usage_metadata", None) or {}

    answer = reply.content.strip()
    found = not any(p in answer.lower() for p in ("couldn't find", "couldn’t find", "could not find"))
    return {
        "answer": answer,
        "found": found,
        "sections": sections if found else [],
        "retrieved_sections": sections,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "model": reply.response_metadata.get("model_name") or reply.response_metadata.get("model"),
    }


def section_link(section: str) -> str:
    return f"{WIKI_URL}#{section.replace(' ', '_')}" if section != "Summary" else WIKI_URL
