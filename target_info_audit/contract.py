from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Channel(str, Enum):
    PREPROCESSING = "preprocessing"
    PROMPTING = "prompting"
    MODEL_INPUT = "model_input"
    POSTPROCESSING = "postprocessing"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


class DependencyKind(str, Enum):
    NO_TARGET = "no_target"
    LABEL_CROP = "label_crop"
    LABEL_SLICE = "label_slice"
    LABEL_NORMALIZATION = "label_normalization"
    LABEL_PROMPT = "label_prompt"
    LABEL_BBOX = "label_bbox"
    LABEL_MASK = "label_mask"
    UNKNOWN = "unknown"


class Authorization(str, Enum):
    DECLARED = "declared"
    UNDECLARED = "undeclared"
    EVALUATION_ONLY = "evaluation_only"
    INCONCLUSIVE = "inconclusive"


class EvidenceType(str, Enum):
    STATIC_SOURCE = "static_source"
    EXECUTED_COUNTERFACTUAL = "executed_counterfactual"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class Status(str, Enum):
    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


@dataclass
class AuditConfig:
    declared_prompt: bool = False
    declared_box: bool = False
    declared_mask: bool = False
    label_used_for_evaluation: bool = True
    allow_prediction_level: bool = False
    seed: int = 0


@dataclass
class PreprocessOutput:
    image_hash: Optional[str] = None
    crop_affine: Optional[List[List[float]]] = None
    crop_origin: Optional[List[int]] = None
    prompt_used: Any = None
    prediction_hash: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditFinding:
    channel: Channel = Channel.UNKNOWN
    kind: DependencyKind = DependencyKind.UNKNOWN
    authorization: Authorization = Authorization.INCONCLUSIVE
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    location: str = ""
    message: str = ""
    confidence: str = "unknown"


@dataclass
class AuditReport:
    pipeline_id: str
    status: Status = Status.APPLICABLE
    findings: List[AuditFinding] = field(default_factory=list)
    declared_budget: Dict[str, bool] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "declared_budget": self.declared_budget,
            "findings": [asdict(f) for f in self.findings],
            "manifest": self.manifest,
        }
