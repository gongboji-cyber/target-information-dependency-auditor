from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .contract import Authorization, Channel, DependencyKind, EvidenceType, AuditFinding


@dataclass
class StaticRule:
    name: str
    pattern: str
    channel: Channel
    kind: DependencyKind
    authorization: Authorization
    note: str
    requires_label_in_same_call: bool = False


RULES: List[StaticRule] = [
    StaticRule(
        "crop_or_pad_label",
        r"CropOrPad\s*\([^)]*mask_name\s*=\s*['\"]label['\"]",
        Channel.PREPROCESSING,
        DependencyKind.LABEL_CROP,
        Authorization.INCONCLUSIVE,
        "Target label controls crop center/padding.",
    ),
    StaticRule(
        "crop_foreground_label",
        r"CropForeground\w*\s*\([^)]*source_key\s*=\s*['\"]label['\"]",
        Channel.PREPROCESSING,
        DependencyKind.LABEL_CROP,
        Authorization.INCONCLUSIVE,
        "Target label controls foreground crop.",
    ),
    StaticRule(
        "label_bbox",
        r"(np\.where\([^\n]*(gt|label|mask)[^\n]*>\s*0\)|gt[^\n]*bbox|label[^\n]*bbox|generate_box\s*\(\s*label|select_points\s*\(\s*label)",
        Channel.PROMPTING,
        DependencyKind.LABEL_BBOX,
        Authorization.INCONCLUSIVE,
        "Label-derived box or point selection; declaration is checked separately.",
    ),
    StaticRule(
        "label_normalization",
        r"(ZNormalization|NormalizeIntensity)[^\n]*masking_method[^\n]*(label|gt)",
        Channel.PREPROCESSING,
        DependencyKind.LABEL_NORMALIZATION,
        Authorization.INCONCLUSIVE,
        "Label mask controls normalization statistics.",
    ),
    StaticRule(
        "label_slice",
        r"(label|gt)[^\n]*(slice|z_|frame_idx|indices)",
        Channel.PREPROCESSING,
        DependencyKind.LABEL_SLICE,
        Authorization.INCONCLUSIVE,
        "Label may control slice/z selection.",
    ),
]


def scan_text(text: str, source_name: str = "") -> List[AuditFinding]:
    findings: List[AuditFinding] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for rule in RULES:
            if re.search(rule.pattern, line, flags=re.IGNORECASE):
                findings.append(
                    AuditFinding(
                        channel=rule.channel,
                        kind=rule.kind,
                        authorization=rule.authorization,
                        evidence_type=EvidenceType.STATIC_SOURCE,
                        location=f"{source_name}:{line_no}",
                        message=f"{rule.note} (pattern: {rule.name})",
                        confidence="candidate",
                    )
                )
    return findings


def scan_file(path: Path) -> List[AuditFinding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return scan_text(text, str(path))


def scan_files(paths: Iterable[Path]) -> List[AuditFinding]:
    out: List[AuditFinding] = []
    for p in paths:
        out.extend(scan_file(p))
    return out
