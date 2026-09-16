"""CPU-friendly, cached Qwen embeddings for hiring proxy models.

Qwen is used as a feature extractor, never as an oracle. This keeps the
experiment's label budget and prohibition on hidden model calls intact.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import duckdb
import numpy as np


@dataclass(frozen=True)
class QwenFeatureConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    dimensions: int = 256
    max_seq_length: int = 384
    max_chars: int = 2400
    batch_size: int = 16

    def validate(self):
        for name in ("dimensions", "max_seq_length", "max_chars", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")


def _compact_resume(parts, max_chars):
    """Keep every available section before spending the remaining char budget."""
    cleaned = [(str(segment), " ".join(str(text).split())) for segment, text in parts if text and str(text).strip()]
    if not cleaned:
        return "[empty resume]"
    per_section = max(48, min(240, max_chars // len(cleaned)))
    heads = [f"[{segment}] {text[:per_section]}" for segment, text in cleaned]
    result = "\n".join(heads)
    if len(result) >= max_chars:
        return result[:max_chars]
    tails = [text[per_section:] for _, text in cleaned]
    cursor = 0
    while len(result) < max_chars and any(tails):
        index = cursor % len(tails)
        piece, tails[index] = tails[index][:256], tails[index][256:]
        if piece:
            result += " " + piece
        cursor += 1
    return result[:max_chars]


def load_resume_texts(database, max_chars=2400):
    """Return stable candidate IDs and compact, section-balanced resume text."""
    grouped = {}
    with duckdb.connect(str(database), read_only=True) as con:
        cursor = con.execute("""
            SELECT candidate_id, segment, text
            FROM candidate_segments
            ORDER BY candidate_id, segment
        """)
        while True:
            rows = cursor.fetchmany(2048)
            if not rows:
                break
            for candidate_id, segment, text in rows:
                grouped.setdefault(candidate_id, []).append((segment, text))
    ids = sorted(grouped)
    if not ids or any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids):
        raise ValueError("candidate_segments must contain nonempty string candidate IDs")
    return ids, [_compact_resume(grouped[candidate_id], max_chars) for candidate_id in ids]


def job_text(job):
    if isinstance(job, (str, Path)):
        job = json.loads(Path(job).read_text(encoding="utf-8"))
    return json.dumps(job, ensure_ascii=False, sort_keys=True)


def _cache_path(database, job, config, cache_dir):
    database = Path(database).resolve()
    stat = database.stat()
    identity = dict(database=str(database), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    job=job_text(job), config=asdict(config), format=1)
    digest = sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(cache_dir or os.environ.get(
        "OKAGENT_QWEN_CACHE", Path.home() / ".cache" / "okagent" / "qwen"))
    return root / f"features-{digest}.npz"


def _default_encoder(config):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install the optional proxy dependencies: pip install -e '.[proxy]'") from error
    encoder = SentenceTransformer(config.model_name, device="cpu")
    encoder.max_seq_length = config.max_seq_length
    return encoder


def _encode(encoder, texts, config, *, query=False):
    values = list(texts)
    if query:
        instruction = "Given a hiring role, retrieve resumes satisfying all must-have qualifications"
        values = [f"Instruct: {instruction}\nQuery: {value}" for value in values]
    vectors = encoder.encode(values, batch_size=config.batch_size, normalize_embeddings=True,
                             show_progress_bar=len(values) > config.batch_size, convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(values) or not np.isfinite(vectors).all():
        raise ValueError("Qwen encoder returned invalid embeddings")
    if vectors.shape[1] < config.dimensions:
        raise ValueError(f"Qwen embedding has only {vectors.shape[1]} dimensions")
    vectors = vectors[:, :config.dimensions]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1)


def load_qwen_features(database, job, *, config=None, cache_dir=None, encoder=None):
    """Load or build reusable candidate and role vectors.

    Returns ``(candidate_ids, vectors, query_vector, metadata)``. The cache is
    atomic and can be shared by baseline, lo_ph, and Hydra runs.
    """
    config = config or QwenFeatureConfig()
    config.validate()
    cache = _cache_path(database, job, config, cache_dir)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            ids = data["candidate_ids"].astype(str).tolist()
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            query = np.asarray(data["query_vector"], dtype=np.float32)
        if vectors.shape == (len(ids), config.dimensions) and query.shape == (config.dimensions,):
            return ids, vectors, query, dict(backend="qwen", cache=str(cache), cached=True, **asdict(config))
    ids, texts = load_resume_texts(database, config.max_chars)
    encoder = encoder or _default_encoder(config)
    vectors = _encode(encoder, texts, config)
    query = _encode(encoder, [job_text(job)], config, query=True)[0]
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, candidate_ids=np.asarray(ids), vectors=vectors, query_vector=query)
    os.replace(temporary, cache)
    return ids, vectors, query, dict(backend="qwen", cache=str(cache), cached=False, **asdict(config))
