from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .contract import AuditReport


def write_reports(reports: List[AuditReport], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for r in reports:
        for f in r.findings:
            rows.append(
                {
                    "pipeline_id": r.pipeline_id,
                    "status": r.status.value,
                    "channel": f.channel.value,
                    "kind": f.kind.value,
                    "authorization": f.authorization.value,
                    "evidence_type": f.evidence_type.value,
                    "location": f.location,
                    "message": f.message,
                    "confidence": f.confidence,
                }
            )
    json_path = out_dir / "audit_reports.json"
    json_path.write_text(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "audit_findings.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("pipeline_id,status,channel,kind,authorization,evidence_type,location,message,confidence\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path}
