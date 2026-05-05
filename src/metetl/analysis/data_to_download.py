"""Подготовка JSON-списка картин для скачивания (Lab 1)."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from metetl.logging_config import get_logger

_log = get_logger("analysis.data_to_download")


def load_paintings_from_csv(csv_path: Path | str) -> List[Dict[str, str]]:
    """Возвращает список словарей {object_id, object_number} для Classification='Paintings'."""
    path = Path(csv_path)
    if not path.exists():
        _log.error("Файл %s не найден", path)
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Classification") == "Paintings":
                object_id = row.get("Object ID")
                if object_id:
                    rows.append({
                        "object_id": object_id,
                        "object_number": row.get("Object Number", ""),
                    })
    _log.info("В CSV найдено %d картин", len(rows))
    return rows


def prepare_to_download(csv_path: Path | str, output_path: Path | str,
                         limit: Optional[int] = None,
                         shuffle: bool = True,
                         seed: Optional[int] = None) -> int:
    """
    Читает CSV, отбирает картины, сохраняет в JSON.
    Возвращает количество записанных картин.
    """
    paintings = load_paintings_from_csv(csv_path)
    if not paintings:
        return 0

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paintings)
    if limit is not None and limit < len(paintings):
        paintings = paintings[:limit]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(paintings, f, ensure_ascii=False, indent=2)

    _log.info("Записано %d картин в %s", len(paintings), out)
    return len(paintings)
