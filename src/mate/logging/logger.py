from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

from mate.logging.utils import ensure_dir


def append_row_csv(path: str | Path, row_dict: Mapping[str, object], fieldnames: Iterable[str]) -> None:
    path = Path(path)
    ensure_dir(path)
    fieldnames = list(fieldnames)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row_dict.get(name, "") for name in fieldnames})


def append_row_jsonl(path: str | Path, row_dict: Dict[str, object]) -> None:
    path = Path(path)
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row_dict, sort_keys=True) + "\n")
