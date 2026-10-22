# Target Information Audit

A minimal Python toolkit for auditing **target-label dependencies** in promptable segmentation inference pipelines. It combines static source-pattern candidates and intervention-based checks, with separate dependency and authorization interpretations.

> **Scope**: The included CLI executes synthetic demonstrations. SAM-Med3D and nnInteractive real-pipeline validation adapters, checkpoints, and medical-image datasets are **not bundled in this repository**. No claim of turnkey reproduction of those GPU validations is made.

## Install

Python 3.9+ is declared; tested environments must be documented separately.

```bash
python -m pip install .
python -m unittest discover -s tests -v
```

## CLI examples

```bash
target-info-audit --adapter label-crop --output-dir report_hidden
target-info-audit --adapter declared-oracle --declared-prompt --output-dir report_oracle
target-info-audit --adapter point-center --output-dir report_clean
```

The CLI generates `audit_reports.json` and `audit_findings.csv`. The built-in synthetic adapters test toy 3D arrays; to audit a real implementation, develop a `PipelineAdapter` calling its preprocessing path and use the Python `run_counterfactual` interface. Static patterns are **candidates**, not evidence of executed or unauthorized use.

## Interpretation

- A changed image/crop can demonstrate label dependence for a tested path.
- Dependence is **not** automatically unauthorized; the evaluation contract must be considered.
- No detected change does not prove absence of all dependencies.
- Prediction differences without stability controls should not be interpreted as causal label propagation.
- Invalid or unsupported paths require human review.

## Reproducibility

`tests/test_core.py` contains lightweight controlled checks. `examples/` contains a version-0.1.0 illustrative output and is not a substitute for a full reproducibility record of real-pipeline experiments.

## License

MIT. No clinical image volumes or third-party model weights are redistributed. See `THIRD_PARTY_NOTICES.md`.
