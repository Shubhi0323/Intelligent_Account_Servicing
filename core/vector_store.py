"""
Vector Store – Semantic Similarity Engine
──────────────────────────────────────────
Uses ChromaDB (in-memory) with a lightweight hash-based embedding function
(no external model downloads required) to compare document text against
known valid templates.

Falls back gracefully to a TF-IDF cosine similarity if ChromaDB is
unavailable.
"""
import logging
import math
import re
from collections import Counter
from core.config import CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)

# ─── Try importing ChromaDB ───────────────────────────────────────────────────
try:
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
    _CHROMA_AVAILABLE = True
    logger.info("ChromaDB loaded successfully.")
except Exception:
    _CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available – falling back to TF-IDF similarity.")

# ─── Template Documents ───────────────────────────────────────────────────────
TEMPLATE_DOCS = {
    "Legal Name Change": [
        "MARRIAGE CERTIFICATE solemnized husband wife spouse registrar married name changed",
        "GAZETTE NOTIFICATION government of india name changed formerly known notification",
        "DIVORCE DECREE court order legal separation name reverting",
    ],
    "Address Change": [
        "ELECTRICITY BILL consumer address utility meter reading due date UPPCL",
        "LEASE AGREEMENT landlord tenant rent address premises agreement notarized",
        "GOVERNMENT ID aadhaar voter identity proof address government",
    ],
    "Date of Birth Change": [
        "BIRTH CERTIFICATE born date of birth municipal registration district",
        "PERMANENT ACCOUNT NUMBER PAN income tax date of birth holder",
        "PASSPORT republic of india date of birth nationality place of birth",
    ],
    "Contact / Email Change": [
        "CONSENT FORM I hereby authorize bank update contact mobile email signature agree",
    ],
}


# ─── Lightweight hash-based embedding (no downloads) ─────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _bag_of_words_embedding(text: str, dim: int = 128) -> list[float]:
    """
    Simple deterministic embedding: each token maps to a bucket via hash.
    Produces a normalised float vector of length `dim`.
    No external model required.
    """
    tokens = _tokenize(text)
    vec = [0.0] * dim
    for token in tokens:
        idx = hash(token) % dim
        vec[idx] += 1.0
    # L2 normalise
    mag = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / mag for x in vec]


# ─── Fallback: lightweight TF-IDF cosine similarity ──────────────────────────

def _tfidf_vector(tokens: list[str], vocab: list[str]) -> list[float]:
    freq = Counter(tokens)
    total = max(sum(freq.values()), 1)
    return [freq.get(w, 0) / total for w in vocab]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _fallback_similarity(text: str, change_type: str) -> float:
    """Cosine similarity against template corpus using TF-IDF bag-of-words."""
    templates = TEMPLATE_DOCS.get(change_type, [])
    if not templates:
        return 0.5

    all_tokens = _tokenize(" ".join(templates) + " " + text)
    vocab = list(set(all_tokens))

    query_vec = _tfidf_vector(_tokenize(text), vocab)
    scores = []
    for tmpl in templates:
        tmpl_vec = _tfidf_vector(_tokenize(tmpl), vocab)
        scores.append(_cosine(query_vec, tmpl_vec))

    return round(max(scores), 4)


# ─── ChromaDB-based similarity ────────────────────────────────────────────────

_chroma_client     = None
_chroma_collection = None


class _HashEmbeddingFunction(chromadb.utils.embedding_functions.EmbeddingFunction):
    """Lightweight embedding function – no model download needed."""
    
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [_bag_of_words_embedding(doc) for doc in input]



def _init_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_client is not None:
        return  # already initialised

    _chroma_client = chromadb.Client()  # ephemeral in-memory
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_HashEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    # Seed with template documents
    docs, ids, metas = [], [], []
    for change_type, templates in TEMPLATE_DOCS.items():
        for i, tmpl in enumerate(templates):
            doc_id = f"{change_type.replace(' ', '_')}_{i}"
            docs.append(tmpl)
            ids.append(doc_id)
            metas.append({"change_type": change_type})

    _chroma_collection.upsert(documents=docs, ids=ids, metadatas=metas)
    logger.info("ChromaDB seeded with %d template documents.", len(docs))


def compute_semantic_similarity(text: str, change_type: str) -> float:
    """
    Return a semantic similarity score (0–1) between the uploaded document
    text and the known valid templates for the given change type.
    """
    if not text or not text.strip():
        return 0.0

    if not _CHROMA_AVAILABLE:
        score = _fallback_similarity(text, change_type)
        logger.info("Fallback TF-IDF similarity=%.4f for %s", score, change_type)
        return score

    try:
        _init_chroma()
        # Filter by change_type if there are matching docs
        templates = TEMPLATE_DOCS.get(change_type, [])
        n_results = min(len(templates), 3) if templates else 1

        results = _chroma_collection.query(
            query_texts=[text],
            n_results=max(n_results, 1),
            where={"change_type": change_type} if templates else None,
        )
        distances = results.get("distances", [[]])[0]
        if not distances:
            return _fallback_similarity(text, change_type)

        # cosine space: distance in [0, 2]; 0 = identical, 2 = opposite
        # Convert to similarity [0, 1]
        best_dist = min(distances)
        score = round(max(1.0 - best_dist / 2.0, 0.0), 4)
        logger.info("ChromaDB similarity=%.4f for %s", score, change_type)
        return score

    except Exception as exc:
        logger.warning("ChromaDB query failed (%s), using fallback.", exc)
        return _fallback_similarity(text, change_type)
