from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .adapters import SyntheticAdapter
from .contract import AuditConfig, AuditReport, Status
from .counterfactual import run_counterfactual
from .report import write_reports
from .static_rules import scan_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Target-information dependency auditor")
    parser.add_argument("--source", type=Path, help="Source file to scan statically")
    parser.add_argument(
        "--adapter",
        choices=["label-crop", "declared-oracle", "point-center"],
        required=True,
    )
    parser.add_argument("--declared-prompt", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    config = AuditConfig(declared_prompt=args.declared_prompt)
    adapter = SyntheticAdapter(args.adapter, {
        "label-crop": "label_crop",
        "declared-oracle": "declared_oracle_prompt",
        "point-center": "point_center",
    }[args.adapter])

    image = np.zeros((8, 8, 8), dtype=np.float32)
    image[2:6, 2:6, 2:6] = 1
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:6, 2:6, 2:6] = 1
    prompt = [4, 4, 4]

    findings = run_counterfactual(adapter, image, prompt, label, config)
    if args.source and args.source.exists():
        findings = scan_file(args.source) + findings
    report = AuditReport(
        pipeline_id=f"synthetic:{args.adapter}",
        status=Status.APPLICABLE,
        findings=findings,
        declared_budget={"declared_prompt": config.declared_prompt},
    )
    paths = write_reports([report], args.output_dir)
    print(json.dumps({"json": str(paths["json"]), "csv": str(paths["csv"]), "findings": len(findings)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
