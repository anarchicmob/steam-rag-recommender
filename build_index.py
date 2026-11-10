#!/usr/bin/env python3
"""
Build the FAISS vector index for Steam games.
Run this once before starting the Flask app:

    uv run python build_index.py

Requires Ollama to be running with nomic-embed-text pulled:
    ollama pull nomic-embed-text

Note: sentence-transformers is not available on Python 3.13 + Intel Mac x86_64
(PyTorch has dropped Intel Mac support). Ollama is used instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
INDEX_DIR = BASE_DIR / "vector_index"

BATCH_SIZE = 100          # rows per SQLite fetch
EMBED_BATCH = 50          # texts per Ollama embed call
MAX_REVIEWS_PER_GAME = 20 # maximum reviews fetched for each game
EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_ollama_embed_fn():
    """Return (embed_fn, model_name, dimension) using Ollama."""
    import ollama
    test = ollama.embed(model=EMBED_MODEL, input="test")
    dim = len(test.embeddings[0])
    print(f"Embedding model: Ollama {EMBED_MODEL} (dim={dim})")

    def embed(texts: list[str]) -> list[list[float]]:
        result = ollama.embed(model=EMBED_MODEL, input=texts)
        return result.embeddings

    return embed, EMBED_MODEL, dim


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_document(row: sqlite3.Row, reviews: list[str]) -> str:
    parts = [f"Game: {row['name']}"]

    if row["short_description"]:
        parts.append(f"Description: {row['short_description']}")

    try:
        genres = json.loads(row["genres_json"] or "[]")
        if genres:
            parts.append(f"Genres: {', '.join(genres)}")
    except Exception:
        pass

    try:
        tags = json.loads(row["tags_json"] or "{}")
        tag_list = list(tags.keys())[:15] if isinstance(tags, dict) else tags[:15]
        if tag_list:
            parts.append(f"Tags: {', '.join(tag_list)}")
    except Exception:
        pass

    if row["release_date"]:
        parts.append(f"Release: {row['release_date']}")

    pos = row["positive"] or 0
    neg = row["negative"] or 0
    if pos + neg > 0:
        pct = round(100 * pos / (pos + neg))
        parts.append(f"Rating: {pct}% positive ({pos + neg} reviews)")

    if reviews:
        snippet = " | ".join(r[:200] for r in reviews[:5])
        parts.append(f"Player reviews: {snippet}")

    return "\n".join(parts)


def fetch_reviews_for_batch(
    conn: sqlite3.Connection, app_ids: list[int]
) -> dict[int, list[str]]:
    placeholders = ",".join("?" * len(app_ids))
    rows = conn.execute(
        f"""
        SELECT appid, review FROM (
            SELECT appid, review,
                   ROW_NUMBER() OVER (
                       PARTITION BY appid ORDER BY LENGTH(review) DESC
                   ) AS rn
            FROM reviews
            WHERE appid IN ({placeholders})
              AND voted_up = 1
              AND review IS NOT NULL
              AND LENGTH(TRIM(review)) > 30
        ) WHERE rn <= {MAX_REVIEWS_PER_GAME}
        """,
        app_ids,
    ).fetchall()
    result: dict[int, list[str]] = {}
    for appid, review in rows:
        result.setdefault(int(appid), []).append(review)
    return result


# ---------------------------------------------------------------------------
# Index build / resume
# ---------------------------------------------------------------------------
def load_existing_index(
    index_path: Path, meta_path: Path
) -> tuple[faiss.Index | None, list[str], dict]:
    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        print(
            f"Resuming: {index.ntotal} games already indexed "
            f"(model={meta.get('model')})"
        )
        return index, meta["app_ids"], meta
    return None, [], {}


def save_checkpoint(index: faiss.Index, app_ids: list[str], meta: dict) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "meta.json").write_text(
        json.dumps({**meta, "app_ids": app_ids}, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Steam game vector index")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max games to process (0 = all, useful for testing)"
    )
    args = parser.parse_args()

    INDEX_DIR.mkdir(exist_ok=True)

    # Embedding function (fail early if Ollama not reachable)
    try:
        embed_fn, model_name, dim = get_ollama_embed_fn()
    except Exception as e:
        sys.exit(
            f"Cannot connect to Ollama: {e}\n"
            "Make sure Ollama is running and you have pulled nomic-embed-text:\n"
            "  ollama pull nomic-embed-text"
        )

    index_path = INDEX_DIR / "index.faiss"
    meta_path = INDEX_DIR / "meta.json"

    existing_index, existing_app_ids, meta = load_existing_index(index_path, meta_path)

    if existing_index is not None and meta.get("model") != model_name:
        sys.exit(
            f"Index was built with model '{meta.get('model')}' "
            f"but current model is '{model_name}'. "
            "Delete vector_index/ and rebuild."
        )

    # Build or extend FAISS index
    if existing_index is not None:
        index = existing_index
    else:
        index = faiss.IndexFlatIP(dim)  # inner product = cosine on L2-normalised vecs

    existing_ids_set = set(existing_app_ids)
    all_app_ids: list[str] = list(existing_app_ids)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute(
        "SELECT COUNT(*) FROM games WHERE name IS NOT NULL AND TRIM(name) != '' AND positive > 100"
    ).fetchone()[0]
    if args.limit:
        total = min(total, args.limit)
    print(f"Total games in DB: {total}  (to process: {total - len(existing_ids_set)})")

    offset = 0
    newly_indexed = 0
    skipped = 0
    errors = 0

    with tqdm(total=total, desc="Indexing", unit="game") as pbar:
        while True:
            rows = conn.execute(
                """
                SELECT appid, name, short_description, genres_json, tags_json,
                       release_date, positive, negative
                FROM games
                WHERE name IS NOT NULL AND TRIM(name) != '' AND positive > 100
                ORDER BY appid
                LIMIT ? OFFSET ?
                """,
                (BATCH_SIZE, offset),
            ).fetchall()

            if not rows:
                break

            if args.limit and offset >= args.limit:
                break

            new_rows = [r for r in rows if str(r["appid"]) not in existing_ids_set]
            skipped += len(rows) - len(new_rows)

            if new_rows:
                app_ids_int = [r["appid"] for r in new_rows]
                reviews_map = fetch_reviews_for_batch(conn, app_ids_int)

                documents = [
                    build_document(row, reviews_map.get(row["appid"], []))
                    for row in new_rows
                ]
                ids = [str(r["appid"]) for r in new_rows]

                # Embed in sub-batches
                sub_vecs: list[list[float]] = []
                for i in range(0, len(documents), EMBED_BATCH):
                    chunk = documents[i : i + EMBED_BATCH]
                    try:
                        sub_vecs.extend(embed_fn(chunk))
                    except Exception as e:
                        print(f"\nEmbed error at offset {offset}+{i}: {e}")
                        errors += len(chunk)

                if sub_vecs:
                    vecs = np.array(sub_vecs, dtype=np.float32)
                    faiss.normalize_L2(vecs) # FAISS need normalized vectors for cosine similarity
                    index.add(vecs)
                    all_app_ids.extend(ids[: len(sub_vecs)])
                    newly_indexed += len(sub_vecs)

                    # Checkpoint every batch
                    save_checkpoint(
                        index,
                        all_app_ids,
                        {"model": model_name, "dimension": dim, "source": "ollama"},
                    )

            pbar.update(len(rows))
            offset += BATCH_SIZE

    conn.close()

    print(f"\n{'=' * 50}")
    print(f"Newly indexed : {newly_indexed}")
    print(f"Skipped       : {skipped}")
    print(f"Errors        : {errors}")
    print(f"Total in index: {index.ntotal}")
    print(f"Embed model   : {model_name}")
    print(f"Index path    : {INDEX_DIR}")


if __name__ == "__main__":
    main()
