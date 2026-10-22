from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_manifest(pipeline_id: str, config: Dict[str, Any], adapter_name: str) -> Dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "adapter": adapter_name,
        "config": config,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }


def write_json(path: Path, obj: Any) -> str:
    data = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)
