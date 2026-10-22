import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from target_info_audit.adapters import SyntheticAdapter  # noqa: E402
from target_info_audit.contract import AuditConfig, Authorization, DependencyKind  # noqa: E402
from target_info_audit.counterfactual import run_counterfactual  # noqa: E402
from target_info_audit.static_rules import scan_text  # noqa: E402


class TestStaticRules(unittest.TestCase):
    def test_label_crop_pattern_detected(self):
        findings = scan_text("crop = tio.CropOrPad(mask_name='label', target_shape=(128,128,128))", "test.py")
        self.assertTrue(any(f.kind == DependencyKind.LABEL_CROP for f in findings))

    def test_clean_snippet_has_no_high_confidence_dependency(self):
        findings = scan_text("crop_mask = create_crop_mask_from_prompt(point)", "clean.py")
        self.assertFalse(any(f.kind == DependencyKind.LABEL_CROP for f in findings))


class TestCounterfactual(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.image = rng.normal(size=(16, 16, 16)).astype(np.float32)
        self.prompt = [8, 8, 8]
        self.label = np.zeros((16, 16, 16), dtype=np.uint8)
        self.label[5:11, 5:11, 5:11] = 1

    def test_hidden_label_crop_detected(self):
        adapter = SyntheticAdapter("hidden", "label_crop")
        findings = run_counterfactual(adapter, self.image, self.prompt, self.label, AuditConfig())
        self.assertTrue(any(f.authorization == Authorization.UNDECLARED for f in findings))

    def test_declared_oracle_not_mislabeled_as_undeclared(self):
        adapter = SyntheticAdapter("oracle", "declared_oracle_prompt")
        findings = run_counterfactual(adapter, self.image, self.prompt, self.label, AuditConfig(declared_prompt=True))
        self.assertTrue(any(f.authorization == Authorization.DECLARED for f in findings))
        self.assertFalse(any(f.authorization == Authorization.UNDECLARED for f in findings))

    def test_declared_mask_crop_not_mislabeled(self):
        adapter = SyntheticAdapter("authorized-roi", "label_crop")
        findings = run_counterfactual(adapter, self.image, self.prompt, self.label, AuditConfig(declared_mask=True))
        self.assertTrue(any(f.authorization == Authorization.DECLARED for f in findings))
        self.assertFalse(any(f.authorization == Authorization.UNDECLARED for f in findings))

    def test_static_pattern_not_automatically_undeclared(self):
        findings = scan_text("tio.CropOrPad(mask_name='label')", "example.py")
        self.assertTrue(findings)
        self.assertTrue(all(f.authorization == Authorization.INCONCLUSIVE for f in findings))


if __name__ == "__main__":
    unittest.main()
