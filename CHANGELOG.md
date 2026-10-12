# Changelog

## Unreleased - core design and saved-result reproduction

### Added

- Shared 83-feature B-prime implementation.
- Five coordinate tests and ten core/model regression tests.
- Seven FoldX parser tests, bringing the current total to 22.
- Saved-output HFE global ranking validation.
- Raw-versus-clipped B-prime metric audit.
- Shared FoldX saved-output parser.
- Production full-precision protein-score consistency analysis.
- Core environment recipes and new-environment validation records.
- Machine-readable result tables and run manifests.

### Fixed

- Removed hard-coded core resource paths.
- Used weighted bystander scores in design-tool Pareto calculation.
- Removed premature rounding from core ranking inputs.
- Replaced broad target exclusion with tested exact coordinate conversion.
- Rejected missing coordinates in exact-exclusion mode.
- Rejected mixed-PAM fallback to NGN.
- Preserved explicitly specified zero ranking weights.
- Merged HFE raw genome-hit scores before global normalization.
- Removed obsolete column-index parsing from the recommended FoldX
  saved-result workflow.
- Removed automatic literature-numbering +1 conversion from revised
  FoldX parser output.
- Preserved full precision in protein-score correlations.
- Replaced historical median-based consensus labels and mixed-scale
  priority scores with explicit source-table nomination records.

### Verified

- Twenty-two tests passed in the new core Conda environment.
- HFE outputs agreed between original and new environments.
- Exact saved-output target exclusion retained 79 HFE and 2074 LMNA hits.
- Global HFE ranking: SpCas9, SpRY, SpCas9-NG.
- Raw B-prime validation/test metrics matched historical values.
- All 44 FoldX means and SDs matched audited saved results.
- Twenty-three of 44 mean energy differences were below zero.
- FoldX SD matched population SD across computational repeats.
- Production protein analysis matched all four audited correlations.
- ESM-IF1 full-precision rho=-0.4786469344608879,
  p=0.0010139508880828937.

### Historical result clarification

- Earlier sequence-heuristic HFE ranking placed SpRY first.
- It is a different stage from global genome-hit ranking.
- Historical ESM-IF1 correlation used scores rounded before analysis.
- Full-precision rho rounds to -0.479 at three decimals.
- Old test logs remain historical evidence; they were not rewritten.

### Known limitations

- Historical conformal calibration reused early-stopping validation data.
- The old FoldX execution entry point remains disabled.
- Generic arbitrary-input parser and search workflows are not fully validated.
- Full retraining, new FoldX/ESM execution and cross-platform reproduction
  remain outside the validated core scope.
- Original code is intended for MIT licensing, but the actual license
  file and third-party/data rights review must be completed before release.
- No independent-user or wet-lab validation is claimed.