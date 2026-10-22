from __future__ import annotations

from typing import Any, Protocol

from .contract import PreprocessOutput


class PipelineAdapter(Protocol):
    """Minimal adapter contract.

    An adapter must either run a preprocessing step (returning image/crop geometry)
    or declare itself unsupported. Prediction-level verification is optional.
    """

    def run_preprocess(self, image: Any, prompt: Any, label: Any, config: Any) -> PreprocessOutput:
        ...

    def run_prediction(self, preprocessed: Any, prompt: Any, config: Any) -> Any:
        ...


class SyntheticAdapter:
    def __init__(self, name: str, behavior: str, seed: int = 0):
        self.name = name
        self.behavior = behavior
        self.seed = seed

    def run_preprocess(self, image, prompt, label, config):
        import hashlib
        import numpy as np

        img = np.asarray(image)
        if self.behavior == "label_crop":
            mask = np.asarray(label) > 0
            coords = np.argwhere(mask)
            origin = [int(c) for c in coords.min(axis=0)] if len(coords) else [0, 0, 0]
            h = hashlib.sha256(np.ascontiguousarray(img[tuple(slice(max(0, o), min(s, o + 8)) for o, s in zip(origin, img.shape))]).tobytes()).hexdigest()
            return PreprocessOutput(image_hash=h, crop_origin=origin, prompt_used=prompt, metadata={"behavior": self.behavior})
        if self.behavior == "point_center":
            origin = [max(0, min(int(p), max(0, s - 8))) for p, s in zip(prompt, img.shape)]
            h = hashlib.sha256(np.ascontiguousarray(img[tuple(slice(o, min(s, o + 8)) for o, s in zip(origin, img.shape))]).tobytes()).hexdigest()
            return PreprocessOutput(image_hash=h, crop_origin=origin, prompt_used=prompt, metadata={"behavior": self.behavior})
        if self.behavior == "declared_oracle_prompt":
            mask = np.asarray(label) > 0
            coords = np.argwhere(mask)
            used_prompt = [int(c) for c in coords.mean(axis=0)] if len(coords) else list(prompt)
            return PreprocessOutput(image_hash=hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest(), prompt_used=used_prompt, metadata={"behavior": self.behavior, "declared": True})
        if self.behavior == "label_eval_only":
            return PreprocessOutput(image_hash=hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest(), prompt_used=prompt, metadata={"behavior": self.behavior})
        raise NotImplementedError(self.behavior)

    def run_prediction(self, preprocessed, prompt, config):
        return None
