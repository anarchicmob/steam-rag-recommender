from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_games_from_sqlite(db_path: Path, limit: int) -> list[tuple[str, dict[str, Any]]]:
    query = """
        SELECT
            appid,
            name,
            short_description,
            about_the_game,
            detailed_description,
            release_date,
            price,
            header_image,
            windows,
            mac,
            linux,
            developers_json,
            publishers_json,
            categories_json,
            genres_json,
            tags_json
        FROM games
        WHERE name IS NOT NULL AND TRIM(name) != ''
        ORDER BY appid
        LIMIT ?
    """

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (limit,)).fetchall()

    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        raw = {
            "name": row["name"] or "Unknown title",
            "short_description": row["short_description"] or "",
            "about_the_game": row["about_the_game"] or "",
            "detailed_description": row["detailed_description"] or "",
            "release_date": row["release_date"],
            "price": row["price"],
            "header_image": row["header_image"],
            "windows": bool(row["windows"]),
            "mac": bool(row["mac"]),
            "linux": bool(row["linux"]),
            "developers": _load_json_value(row["developers_json"], []),
            "publishers": _load_json_value(row["publishers_json"], []),
            "categories": _load_json_value(row["categories_json"], []),
            "genres": _load_json_value(row["genres_json"], []),
            "tags": _load_json_value(row["tags_json"], {}),
        }
        records.append((str(row["appid"]), raw))

    return records


def _load_json_value(payload: str | None, default: Any) -> Any:
    if not payload:
        return default

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default
