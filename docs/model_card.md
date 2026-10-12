# B-prime model card

## Identity

B-prime (B′): ABE8e guide-sequence efficiency regression model.

Current local asset:

```text
models/xgboost_design_v2.pkl
```

Framework: XGBoost.

The model is intended for research candidate evaluation, not clinical
decision-making or autonomous therapeutic design.

## Training context

- Data source: BE-dataHIVE.
- Core study: study_id=3.
- Editor: ABE8e.
- Cell line: HEK293T.
- Target label: efficiency_full_grna_reported.
- Fixed dataset sizes:
  - Train: 15530.
  - Validation: 3328.
  - Test: 3329.

B-prime reused the full-feature comparison dataset, including removal of
rows with missing energy features, to preserve comparability.
B-prime itself does not use those 24 energy features.

## Input representation

Exactly 20 A/C/G/T bases.

Feature vector:

1. 80 positional one-hot values, base order A,T,G,C.
2. ViennaRNA MFE of the spacer treated as RNA.
3. Biopython nearest-neighbor Tm of the spacer.
4. The same Tm again for historical model compatibility.

Total: 83 columns.

The duplicate Tm is intentional compatibility with the trained artifact,
not an independent physical measurement.

Do not remove or reorder features without training and validating a
new model version.

Shared implementation:

```text
scripts/bprime_features.py
```

## Output policy

Historical evaluation reports raw regression output.

The design tool clips predicted efficiency to [0,1].

Test metrics:

| Policy | Spearman | MAE |
|---|---:|---:|
| Raw | 0.670049192932404 | 0.10876274617229263 |
| Clipped | approximately 0.6700606317 | approximately 0.1085982630 |

The shared feature implementation reproduced historical raw validation
and test metrics in the recorded audit.

Prediction is an estimated label value, not a guaranteed target-base
editing frequency in a new assay.

## Data splitting

The original script groups by exact full_context_sequence values.
This does not automatically prove the absence of every form of
sequence-similarity, locus or guide-level overlap.

The original split depends on the input snapshot and ordering.
Preserve fixed split files and hashes when reproducing historical metrics.

## Uncertainty intervals

Historical conformal residuals were calibrated on the same validation
split used for early stopping.

Recorded test empirical coverage:

- 80% interval: 0.8005407029137879.
- 90% interval: 0.8954641033343347.

These are empirical results. Standard independent-calibration coverage
guarantees are not claimed for this setup.

## Applicability limits

- Cross-study and cross-cell transfer are not established.
- Cas-variant-specific efficiency calibration is not established.
- PAM enumeration does not itself validate editor functionality.
- Efficiency prediction does not capture all bystander or off-target effects.
- No clinical performance claim is made.
- No wet-lab validation of newly proposed candidates is claimed.

## Artifact integrity

The recorded model SHA256 is available in:

```text
results/release_validation/bprime_prediction_policy/run_manifest.json
```

Check the corresponding artifact hash after downloading or moving assets.

Pickle loading can execute code. Load only trusted artifacts.
Do not substitute another model under the same filename without updating
metadata and rerunning tests.

## License

Original code is planned for MIT licensing, subject to the actual LICENSE
file and confirmed rights.

Model-asset distribution rights and underlying data conditions must be
reviewed separately. The model filename alone does not establish a license.