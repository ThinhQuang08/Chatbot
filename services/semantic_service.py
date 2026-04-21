# services/semantic_service.py
import math
import logging
import uuid
import time
from functools import lru_cache
from hashlib import sha1
from typing import Dict, Iterable, List, Optional

from config.settings import (
    ENABLE_SEMANTIC_SEARCH,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    SEMANTIC_BACKEND,
    SEMANTIC_MODEL_NAME,
)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models

    _qdrant_import_error: Optional[str] = None
except Exception as exc:
    QdrantClient = None
    qdrant_models = None
    _qdrant_import_error = str(exc)


_semantic_model: Optional[object] = None
_qdrant_client: Optional[object] = None

# _collection_exists_cache: Optional[bool] = None
# _collection_exists_cache_at: float = 0.0
_collection_exists_cache: Dict[str, bool] = {}
_collection_exists_cache_at: Dict[str, float] = {}

_collection_missing_warned: bool = False
_COLLECTION_CACHE_TTL_SECONDS = 30.0

logger = logging.getLogger(__name__)


def _semantic_enabled() -> bool:
    return str(ENABLE_SEMANTIC_SEARCH).lower() in {"1", "true", "yes"}


def _backend() -> str:
    return str(SEMANTIC_BACKEND or "local").strip().lower()


def semantic_available() -> bool:
    return SentenceTransformer is not None and _semantic_enabled()


def _get_model() -> Optional[object]:
    global _semantic_model
    if SentenceTransformer is None:
        return None
    if _semantic_model is None:
        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    return _semantic_model


def _get_qdrant_client() -> Optional[object]:
    global _qdrant_client
    if _backend() != "qdrant":
        return None
    if QdrantClient is None:
        return None
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=3.0,
        )
    return _qdrant_client


@lru_cache(maxsize=4096)
def _embed_cached(text: str) -> Optional[tuple[float, ...]]:
    model = _get_model()
    if model is None:
        return None
    vector = model.encode(text, show_progress_bar=False)
    return tuple(float(value) for value in vector)


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left))
    right_norm = math.sqrt(sum(r * r for r in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def _qdrant_point_id(candidate_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, candidate_id))


# def _ensure_qdrant_collection(client: object, vector_size: int) -> None:
#     if qdrant_models is None:
#         return

#     collections = client.get_collections()
#     names = {collection.name for collection in collections.collections}
#     if QDRANT_COLLECTION in names:
#         return

#     client.create_collection(
#         collection_name=QDRANT_COLLECTION,
#         vectors_config=qdrant_models.VectorParams(
#             size=vector_size,
#             distance=qdrant_models.Distance.COSINE,
#         ),
#     )


# def _collection_exists(client: object) -> bool:
#     collections = client.get_collections()
#     names = {collection.name for collection in collections.collections}
#     return QDRANT_COLLECTION in names


# def _collection_exists_cached(client: object, force: bool = False) -> bool:
#     global _collection_exists_cache, _collection_exists_cache_at

#     now = time.time()
#     if (
#         not force
#         and _collection_exists_cache is not None
#         and (now - _collection_exists_cache_at) < _COLLECTION_CACHE_TTL_SECONDS
#     ):
#         return _collection_exists_cache

#     exists = _collection_exists(client)
#     _collection_exists_cache = exists
#     _collection_exists_cache_at = now
#     return exists

def _ensure_qdrant_collection(client: object, vector_size: int, collection_name: str = QDRANT_COLLECTION) -> None:
    if qdrant_models is None:
        return

    collections = client.get_collections()
    names = {collection.name for collection in collections.collections}
    if collection_name in names:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )

def _collection_exists(client: object, collection_name: str = QDRANT_COLLECTION) -> bool:
    collections = client.get_collections()
    names = {collection.name for collection in collections.collections}
    return collection_name in names


def _collection_exists_cached(client: object, force: bool = False, collection_name: str = QDRANT_COLLECTION) -> bool:
    global _collection_exists_cache, _collection_exists_cache_at

    now = time.time()
    if (
        not force
        and collection_name in _collection_exists_cache
        and (now - _collection_exists_cache_at.get(collection_name, 0)) < _COLLECTION_CACHE_TTL_SECONDS
    ):
        return _collection_exists_cache[collection_name]

    exists = _collection_exists(client, collection_name)
    _collection_exists_cache[collection_name] = exists
    _collection_exists_cache_at[collection_name] = now
    return exists


def build_destination_candidate_id(
    location: Optional[str],
    season: Optional[str],
    description: Optional[str],
    activities: Optional[str],
) -> str:
    raw = "|".join(
        [
            str(location or ""),
            str(season or ""),
            str(description or ""),
            str(activities or ""),
        ]
    )
    return sha1(raw.encode("utf-8")).hexdigest()


def build_destination_semantic_text(
    location: Optional[str],
    description: Optional[str],
    activities: Optional[str],
    price_level: Optional[str],
    season: Optional[str],
) -> str:
    return " ".join(
        [
            str(location or "").lower(),
            str(description or "").lower(),
            str(activities or "").lower(),
            str(price_level or "").lower(),
            str(season or "").lower(),
        ]
    )


def qdrant_healthcheck() -> Dict[str, object]:
    if _backend() != "qdrant":
        return {
            "ok": False,
            "backend": _backend(),
            "reason": "SEMANTIC_BACKEND is not qdrant",
        }

    client = _get_qdrant_client()
    if client is None:
        return {
            "ok": False,
            "backend": "qdrant",
            "reason": "qdrant-client is not available",
            "import_error": _qdrant_import_error,
        }

    try:
        exists = _collection_exists(client)
        return {
            "ok": True,
            "backend": "qdrant",
            "collection": QDRANT_COLLECTION,
            "collection_exists": exists,
            "url": QDRANT_URL,
        }
    except Exception as exc:
        return {
            "ok": False,
            "backend": "qdrant",
            "reason": str(exc),
            "collection": QDRANT_COLLECTION,
            "url": QDRANT_URL,
        }


# def sync_qdrant_documents(
#     documents: Iterable[Dict[str, object]],
#     recreate_collection: bool = False,
#     batch_size: int = 128,
# ) -> Dict[str, object]:
#     global _collection_exists_cache, _collection_exists_cache_at, _collection_missing_warned

#     if _backend() != "qdrant":
#         return {
#             "ok": False,
#             "reason": "SEMANTIC_BACKEND must be qdrant to sync",
#         }

#     model = _get_model()
#     client = _get_qdrant_client()
#     if model is None or client is None or qdrant_models is None:
#         return {
#             "ok": False,
#             "reason": "Missing SentenceTransformer or qdrant-client",
#         }

#     docs = list(documents)
#     if not docs:
#         return {"ok": True, "synced": 0, "collection": QDRANT_COLLECTION}

#     try:
#         first_text = str(docs[0].get("text") or "")
#         sample_vector = model.encode(first_text, show_progress_bar=False)
#         vector_size = len(sample_vector)

#         if recreate_collection and _collection_exists(client):
#             client.delete_collection(collection_name=QDRANT_COLLECTION)

#         _ensure_qdrant_collection(client, vector_size=vector_size)

#         synced = 0
#         for index in range(0, len(docs), max(batch_size, 1)):
#             chunk = docs[index : index + max(batch_size, 1)]
#             points = []
#             for item in chunk:
#                 candidate_id = str(item.get("candidate_id") or "")
#                 text = str(item.get("text") or "")
#                 if not candidate_id or not text:
#                     continue
#                 payload = dict(item.get("payload") or {})
#                 payload["candidate_id"] = candidate_id
#                 vector = model.encode(text, show_progress_bar=False)
#                 points.append(
#                     qdrant_models.PointStruct(
#                         id=_qdrant_point_id(candidate_id),
#                         vector=[float(value) for value in vector],
#                         payload=payload,
#                     )
#                 )

#             if points:
#                 client.upsert(
#                     collection_name=QDRANT_COLLECTION,
#                     points=points,
#                     wait=True,
#                 )
#                 synced += len(points)

#             _collection_exists_cache = True
#             _collection_exists_cache_at = time.time()
#             _collection_missing_warned = False

#         return {
#             "ok": True,
#             "synced": synced,
#             "collection": QDRANT_COLLECTION,
#             "url": QDRANT_URL,
#         }
#     except Exception as exc:
#         logger.warning("Qdrant sync failed: %s", exc)
#         return {
#             "ok": False,
#             "reason": str(exc),
#             "collection": QDRANT_COLLECTION,
#             "url": QDRANT_URL,
#         }

def sync_qdrant_documents(
    documents: Iterable[Dict[str, object]],
    recreate_collection: bool = False,
    batch_size: int = 128,
    collection_name: str = QDRANT_COLLECTION,  # MỚI: Thêm tham số này
) -> Dict[str, object]:
    global _collection_exists_cache, _collection_exists_cache_at, _collection_missing_warned

    if _backend() != "qdrant":
        return {
            "ok": False,
            "reason": "SEMANTIC_BACKEND must be qdrant to sync",
        }

    model = _get_model()
    client = _get_qdrant_client()
    if model is None or client is None or qdrant_models is None:
        return {
            "ok": False,
            "reason": "Missing SentenceTransformer or qdrant-client",
        }

    docs = list(documents)
    if not docs:
        return {"ok": True, "synced": 0, "collection": collection_name}

    try:
        first_text = str(docs[0].get("text") or "")
        sample_vector = model.encode(first_text, show_progress_bar=False)
        vector_size = len(sample_vector)

        if recreate_collection and _collection_exists(client, collection_name):
            client.delete_collection(collection_name=collection_name)

        _ensure_qdrant_collection(client, vector_size=vector_size, collection_name=collection_name)

        synced = 0
        for index in range(0, len(docs), max(batch_size, 1)):
            chunk = docs[index : index + max(batch_size, 1)]
            points = []
            for item in chunk:
                candidate_id = str(item.get("candidate_id") or "")
                text = str(item.get("text") or "")
                if not candidate_id or not text:
                    continue
                payload = dict(item.get("payload") or {})
                payload["candidate_id"] = candidate_id
                vector = model.encode(text, show_progress_bar=False)
                points.append(
                    qdrant_models.PointStruct(
                        id=_qdrant_point_id(candidate_id),
                        vector=[float(value) for value in vector],
                        payload=payload,
                    )
                )

            if points:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                synced += len(points)

            _collection_exists_cache[collection_name] = True
            _collection_exists_cache_at[collection_name] = time.time()
            _collection_missing_warned = False

        return {
            "ok": True,
            "synced": synced,
            "collection": collection_name,
            "url": QDRANT_URL,
        }
    except Exception as exc:
        logger.warning("Qdrant sync failed: %s", exc)
        return {
            "ok": False,
            "reason": str(exc),
            "collection": collection_name,
            "url": QDRANT_URL,
        }

def semantic_scores(query: str, texts_by_id: Dict[str, str]) -> Dict[str, float]:
    if not _semantic_enabled() or not query.strip() or not texts_by_id:
        return {candidate_id: 0.0 for candidate_id in texts_by_id}

    if _backend() == "qdrant":
        scores = _semantic_scores_qdrant(query, texts_by_id)
        if scores is not None:
            return scores

    return _semantic_scores_local(query, texts_by_id)


def _semantic_scores_local(query: str, texts_by_id: Dict[str, str]) -> Dict[str, float]:
    query_vector = _embed_cached(query)
    if query_vector is None:
        return {candidate_id: 0.0 for candidate_id in texts_by_id}

    scores: Dict[str, float] = {}
    for candidate_id, text in texts_by_id.items():
        text_vector = _embed_cached(text)
        if text_vector is None:
            scores[candidate_id] = 0.0
            continue
        scores[candidate_id] = _cosine_similarity(list(query_vector), list(text_vector))
    return scores


def _semantic_scores_qdrant(
    query: str,
    texts_by_id: Dict[str, str],
) -> Optional[Dict[str, float]]:
    global _collection_missing_warned

    model = _get_model()
    client = _get_qdrant_client()

    if model is None or client is None or qdrant_models is None:
        return None

    try:
        query_vector = model.encode(query, show_progress_bar=False)
        query_list = [float(value) for value in query_vector]
        if not _collection_exists_cached(client):
            if not _collection_missing_warned:
                logger.warning(
                    "Qdrant collection '%s' does not exist. Run sync script first.",
                    QDRANT_COLLECTION,
                )
                _collection_missing_warned = True
            return {candidate_id: 0.0 for candidate_id in texts_by_id}

        _collection_missing_warned = False

        candidate_ids = list(texts_by_id.keys())

        try:
            hits = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_list,
                query_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="candidate_id",
                            match=qdrant_models.MatchAny(any=candidate_ids),
                        )
                    ]
                ),
                limit=len(candidate_ids),
                with_payload=True,
            )

            # log xem Qdrant search
            for hit in hits:
                print("QDRANT RESULT:", hit.payload["location"], hit.score)
                
        except Exception:
            hits = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_list,
                limit=max(len(candidate_ids) * 3, 50),
                with_payload=True,
            )

        scores = {candidate_id: 0.0 for candidate_id in texts_by_id}
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            candidate_id = payload.get("candidate_id")
            if candidate_id in scores:
                scores[candidate_id] = float(getattr(hit, "score", 0.0) or 0.0)
        return scores
    except Exception as exc:
        logger.warning("Qdrant semantic search failed, fallback to local: %s", exc)
        return None


def semantic_similarity(query: str, text: str) -> float:
    if not query.strip() or not text.strip() or not _semantic_enabled():
        return 0.0

    scores = semantic_scores(query, {"_single": text})
    return scores.get("_single", 0.0)


# embedding tour
def build_tour_candidate_id(tour_id: str, chunk_type: str = "overview", chunk_index: int = 0) -> str:
    """Tạo ID duy nhất cho mỗi chunk của Tour"""
    raw = f"{tour_id}|{chunk_type}|{chunk_index}"
    return sha1(raw.encode("utf-8")).hexdigest()

def build_tour_semantic_text(
    tour_name: str, 
    destinations: List[str], 
    description: str
) -> str:
    """Tạo đoạn văn bản để SBERT đọc hiểu và nhúng Vector"""
    dests = ", ".join(destinations) if destinations else ""
    return " ".join(
        [
            str(tour_name or "").lower(),
            str(dests or "").lower(),
            str(description or "").lower(),
        ]
    )