# Validation status

## Release scope

This repository's validated release scope is:

**Core ABE8e candidate design and saved-result reproduction.**

It is not a claim that every historical research script has been audited
or that the entire study has been rerun from scratch.

## Environment validation

Completed on the author's Windows machine in a newly created Conda
environment named `abe8e-core-check`:

- Python 3.10.20.
- Pinned core direct dependencies installed successfully.
- `python -m pip check` reported no broken requirements.
- HFE candidate identities, coordinates, Pareto flags and numerical
  outputs agreed with the original TZS environment.
- The current 22 automatic tests passed.

This is new-environment validation on the same machine, not another-machine,
cross-platform or independent-user reproduction.

## Latest test evidence

Command:

```bat
python -m unittest discover -s tests -p "test_*.py" -v
```

Latest recorded log:

```text
results/release_validation/release_core/tests_22.txt
```

The user-observed process exit code was 0.

Earlier 15-test logs remain historical evidence and must not be edited
to imply that 22 tests existed at that time.

## Verified computations

### B-prime inference

- Shared 83-feature recipe used by the revised core code.
- HFE example generated three candidates.
- Fixed-sequence model regression test passed.
- Historical raw validation and test metrics were reproduced with
  zero difference in the recorded metric audit.

### Raw versus clipped prediction

Test raw predictions:

- Spearman: 0.670049192932404
- MAE: 0.10876274617229263

Test predictions clipped to [0,1]:

- Spearman: approximately 0.6700606317
- MAE: approximately 0.1085982630

There were 90 negative validation predictions and 108 negative test
predictions. Neither split had predictions above 1.

### Saved genome-search outputs

- Exact coordinate conversion was checked for plus and minus strands.
- HFE SpCas9: 79 remaining hit records after excluding one intended target.
- LMNA VRQR: 2074 remaining hit records after excluding one intended target.
- HFE three-candidate outputs passed line-count, PAM, mismatch-count
  and duplicate-record checks in the dedicated validation workflow.
- Global HFE normalization and scoring were performed after merging
  the three PAM-specific result sets.
- Ranking under weights 0.4/0.3/0.3: SpCas9, SpRY, SpCas9-NG.
- SpCas9 and SpRY are Pareto non-dominated.

These checks reanalysed saved outputs; they did not rerun the genome search.

### Saved FoldX results

- Forty-four unique substitutions matched the candidate input.
- Five computational repeats per substitution were checked.
- Raw mutant-minus-matched-WT values were checked against Dif.
- Dif means were checked against Average.
- Reported SD matched population SD across repeats, ddof=0.
- All 44 mean energy differences and SDs were reproduced by the
  shared production parser.
- Twenty-three of 44 mean energy differences were below zero.

### Protein-model consistency

- Three saved score matrices had shape (166,20) and finite entries.
- Candidate scores matched matrix-derived scores within the specified
  absolute tolerance.
- The production analysis reproduced all four audited full-precision
  correlations.
- ESM-IF1: rho=-0.4786469344608879, p=0.0010139508880828937.
- P-values are two-sided and unadjusted across the four correlations.

## Current entry-point status

| Entry point | Status |
|---|---|
| design_sgrna.py | HFE core example validated |
| validate_bprime_prediction_policy.py | Fixed-data metric audit validated |
| validate_hfe_global_ranking.py | Saved HFE output reanalysis validated |
| parse_foldx_ddg.py | Shared strict parser validated |
| ddg_lm_consistency.py | Full-precision saved-score analysis validated |
| foldx_ddg.py | Historical execution entry point remains disabled |
| Remaining training/protein/exploratory scripts | Not all fully validated |

## Important limitations

- No wet-lab functional or safety validation is claimed.
- Genome-hit and bystander scores are not calibrated biological probabilities.
- Cross-Cas efficiency calibration has not been established.
- Historical conformal calibration reused the early-stopping validation split.
- Protein SD represents computational-repeat variation, not experimental
  uncertainty or total structural-model uncertainty.
- Structural methods share predicted-structure input.
- Full model retraining and new FoldX/ESM execution have not been validated
  as part of the core release.
- Public redistribution permissions and final asset packaging still require review.