# Saved protein-result validation

## Status

Both stages are now validated:

1. Production parsing through parse_foldx_ddg.py and foldx_results.py.
2. Full-precision correlation analysis through ddg_lm_consistency.py.

The historical new-computation entry point foldx_ddg.py remains disabled.

No FoldX computation or ESM model inference was rerun in this validation.

## FoldX output checks

- 44 unique substitutions.
- 5 computational repeats per substitution.
- Complete agreement between the mutation list and candidate table.
- Model identifiers checked rather than relying only on row order.
- Raw mutant-minus-matched-WT energies compared with Dif.
- Dif repeat means compared with Average.
- Original three-decimal results compared with full-precision output.

Recorded maximum discrepancies:

| Comparison | Maximum absolute discrepancy |
|---|---:|
| Raw mutant-WT versus Dif | 9.200000000009201e-05 |
| Average versus mean of Dif | 5.999999999950489e-06 |
| Historical rounded mean versus Average | 0.0004959999999999964 |
| Historical rounded SD versus Average SD | 0.0004597000000000004 |
| Reported SD versus population SD | 2.2161729875055936e-06 |
| Reported SD versus sample SD | 0.04551731965083944 |

The numerical differences are consistent with the recorded text-output
precision checks.

The SD convention is population SD across computational repeats, ddof=0.
This SD is not experimental uncertainty or total model uncertainty.

## Stable-sign count

23 of 44 reported mean energy differences are below zero.

This is a sign-based summary, not proof that all 23 substitutions produce
meaningful functional or stability improvements.

Near-zero examples require caution:

- T132V: mean approximately -0.003031.
- F83L: mean approximately +0.017003.

Do not turn a sign-based label into a strong biological conclusion.

## Candidate terminology

candidate_consensus_exact.csv contains a merged set of 44 substitutions.
Some were nominated by only one model.

It must not be described as 44 strict multi-model consensus mutations.

Nomination counts in the revised consistency workflow are checked against
the three source candidate tables.

## Matrix validation

Three matrices had shape (166,20), float32 values and finite entries.

Candidate-table comparisons:

| Model | Candidates checked | Maximum absolute error | Tolerance |
|---|---:|---:|---:|
| ESM2 | 15 | approximately 0.000466 | 0.01 |
| ESM1v | 15 | approximately 0.000476 | 0.01 |
| ESM-IF1 | 20 | approximately 0.000483 | 0.01 |

These are internal consistency checks within a tolerance, not proof of
independent experimental correctness.

## Full-precision correlations

| Score | Spearman | p-value |
|---|---:|---:|
| ESM2 | -0.03847780126849894 | 0.8041522826025486 |
| ESM1v | 0.2545454545454545 | 0.09542538956013835 |
| Sequence-score mean | 0.21916842847075405 | 0.1528916114627122 |
| ESM-IF1 | -0.4786469344608879 | 0.0010139508880828937 |

P-values are two-sided and unadjusted across the four reported correlations.

Recommended rounded presentation:

**ESM-IF1: Spearman rho=-0.479, p approximately 0.001, n=44.**

Historical three-decimal score processing yielded:

- rho=-0.47845237705450117
- p=0.0010194590342722488

Both historical and revised results should remain traceable.

## Interpretation boundaries

- Results are conditional on the selected candidate set.
- Structural methods share predicted-structure input.
- Correlation is not independent functional validation.
- Failure to detect sequence-score correlation does not prove independence.
- The sequence-score mean is an exploratory, uncalibrated aggregation.
- The revised workflow no longer produces the old median-based consensus
  label or mixed-scale priority score.
- No automatic literature-numbering +1 conversion is performed.

## Verified output locations

Production parser:

- results/release_validation/foldx_parser_release/

Production consistency analysis:

- results/release_validation/protein_consistency_release/

Development audit:

- results/release_validation/protein_saved_audit/

See each run_manifest.json or audit_report.json for recorded checksums.