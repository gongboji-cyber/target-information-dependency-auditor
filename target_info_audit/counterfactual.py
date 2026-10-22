from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np

from .contract import (
    AuditConfig,
    AuditFinding,
    Authorization,
    Channel,
    DependencyKind,
    EvidenceType,
    PreprocessOutput,
)


def _sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def make_label_variants(label: Any, rng: np.random.Generator) -> Dict[str, Any]:
    lab = np.asarray(label)
    variants = {"true": lab.copy()}
    if lab.ndim == 3:
        variants["shifted"] = np.roll(lab, shift=8, axis=(0, 1, 2))
        variants["random"] = (rng.random(lab.shape) < (lab > 0).mean()).astype(lab.dtype)
        variants["empty"] = np.zeros_like(lab)
    else:
        variants["shifted"] = np.roll(lab, shift=1, axis=0)
        variants["random"] = (rng.random(lab.shape) < (lab > 0).mean()).astype(lab.dtype)
        variants["empty"] = np.zeros_like(lab)
    return variants


def run_counterfactual(
    adapter,
    image: Any,
    prompt: Any,
    label: Any,
    config: AuditConfig,
) -> List[AuditFinding]:
    rng = np.random.default_rng(config.seed)
    variants = make_label_variants(label, rng)
    outputs: Dict[str, PreprocessOutput] = {}
    for name, lab in variants.items():
        try:
            outputs[name] = adapter.run_preprocess(image, prompt, lab, config)
        except Exception as exc:  # noqa: BLE001
            outputs[name] = PreprocessOutput(warnings=[f"{name}: {type(exc).__name__}: {exc}"])

    base = outputs.get("true")
    if base is None:
        return []

    findings: List[AuditFinding] = []
    changed_variants = []
    for name in ["shifted", "random", "empty"]:
        out = outputs.get(name)
        if out is None:
            continue
        image_changed = (out.image_hash is not None and base.image_hash is not None and out.image_hash != base.image_hash)
        origin_changed = (out.crop_origin is not None and base.crop_origin is not None and out.crop_origin != base.crop_origin)
        prompt_changed = (out.prompt_used is not None and base.prompt_used is not None and not np.array_equal(np.asarray(out.prompt_used), np.asarray(base.prompt_used)))
        if image_changed or origin_changed or prompt_changed:
            changed_variants.append(name)

    image_or_origin_changed = any(v in changed_variants for v in changed_variants) and (
        any(outputs[v].image_hash is not None and base.image_hash is not None and outputs[v].image_hash != base.image_hash for v in changed_variants)
        or any(outputs[v].crop_origin is not None and base.crop_origin is not None and outputs[v].crop_origin != base.crop_origin for v in changed_variants)
    )
    prompt_changed = any(
        v in changed_variants and not np.array_equal(np.asarray(outputs[v].prompt_used), np.asarray(base.prompt_used))
        for v in changed_variants
    )

    if image_or_origin_changed:
        findings.append(
            AuditFinding(
                channel=Channel.PREPROCESSING,
                kind=DependencyKind.LABEL_CROP,
                authorization=(Authorization.DECLARED if config.declared_mask else Authorization.UNDECLARED),
                evidence_type=EvidenceType.EXECUTED_COUNTERFACTUAL,
                location="counterfactual",
                message=f"Preprocessed output changed under label variants: {changed_variants}.",
                confidence="executed",
            )
        )
    elif prompt_changed and (config.declared_prompt or config.declared_box):
        findings.append(
            AuditFinding(
                channel=Channel.PROMPTING,
                kind=DependencyKind.LABEL_PROMPT,
                authorization=Authorization.DECLARED,
                evidence_type=EvidenceType.EXECUTED_COUNTERFACTUAL,
                location="counterfactual",
                message="Declared prompt changed under label substitution; no undeclared preprocessing change.",
                confidence="executed",
            )
        )
    elif prompt_changed:
        findings.append(
            AuditFinding(
                channel=Channel.PROMPTING,
                kind=DependencyKind.LABEL_PROMPT,
                authorization=Authorization.UNDECLARED,
                evidence_type=EvidenceType.EXECUTED_COUNTERFACTUAL,
                location="counterfactual",
                message="Prompt changed under label substitution without declaration.",
                confidence="executed",
            )
        )

    if config.allow_prediction_level:
        try:
            preds = {n: adapter.run_prediction(outputs[n], prompt, config) for n in outputs}
            pred_hashes = {n: _sha(p) if p is not None else None for n, p in preds.items()}
            if len({h for h in pred_hashes.values() if h is not None}) > 1:
                findings.append(
                    AuditFinding(
                        channel=Channel.MODEL_INPUT,
                        kind=DependencyKind.UNKNOWN,
                        authorization=Authorization.INCONCLUSIVE,
                        evidence_type=EvidenceType.EXECUTED_COUNTERFACTUAL,
                        location="prediction",
                        message="Prediction output changed under label substitution.",
                        confidence="executed",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            findings.append(
                AuditFinding(
                    channel=Channel.MODEL_INPUT,
                    kind=DependencyKind.UNKNOWN,
                    authorization=Authorization.INCONCLUSIVE,
                    evidence_type=EvidenceType.UNKNOWN,
                    location="prediction",
                    message=f"Prediction-level check failed: {type(exc).__name__}: {exc}",
                    confidence="inconclusive",
                )
            )
    return findings
