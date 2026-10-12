# Result and reproduction index

Run commands from the repository root in the core Conda environment.
Commands below use Windows Anaconda Prompt syntax.

Use a new output directory for each validation run. Do not overwrite
historical evidence to make it resemble a revised result.

## 1. Automatic tests

```bat
python -m unittest discover -s tests -p "test_*.py" -v
```

Test files:

- tests/test_core_pipeline.py
- tests/test_offtarget_coordinates.py
- tests/test_foldx_results.py

Current expected total: 22 passing tests.

Latest recorded evidence:

- results/release_validation/release_core/tests_22.txt

## 2. HFE sequence-heuristic design

Entry point:

- scripts/design_sgrna.py

Validated outputs:

- results/release_validation/hfe_candidates_patched.csv
- results/release_validation/hfe_candidates_clean_env.csv

These use weighted bystander scores and a sequence-heuristic off-target
score, not genome-hit scores.

The historical heuristic ranking placed SpRY first.

## 3. Exact intended-target exclusion

Entry point:

- scripts/run_true_offtarget.py, using saved output and --summary_only

Validated summaries:

- results/release_validation/hfe_spcas9_exact/offtarget_summary_by_grna.csv
- results/release_validation/lmna_vrqr_exact/offtarget_summary_by_grna.csv

Remaining hit records:

- HFE SpCas9: 79.
- LMNA VRQR: 2074.

One expected target record was excluded in each case, including the
reference-versus-variant mismatch.

Search conditions: corresponding NGG/NGA PAM rules and at most four
mismatches, using the saved hg38 search outputs.

## 4. HFE global genome-hit ranking

Entry point:

- scripts/validate_hfe_global_ranking.py

Command using the already validated candidate table:

```bat
python scripts\validate_hfe_global_ranking.py --candidates results\release_validation\hfe_candidates_patched.csv --outdir results\user_runs\hfe_global
```

Required saved search data:

- results/hfe_offtarget_real/run_SpCas9/
- results/hfe_offtarget_real/run_SpCas9NG/
- results/hfe_offtarget_real/run_SpRY/

Each run needs casoffinder_input.txt and casoffinder_output.txt.

Validated outputs:

- results/release_validation/hfe_global_exact/combined_summary.csv
- results/release_validation/hfe_global_exact/candidates_ranked_global.csv
- results/release_validation/hfe_global_exact/run_manifest.json

Order under weights 0.4/0.3/0.3:

1. SpCas9
2. SpRY
3. SpCas9-NG

Normalization occurs once across the combined candidate set.
Scores are not directly comparable across independently normalized loci.

## 5. B-prime metric policy audit

Entry point:

- scripts/validate_bprime_prediction_policy.py

```bat
python scripts\validate_bprime_prediction_policy.py --outdir results\user_runs\bprime_policy
```

Required assets:

- models/xgboost_design_v2.pkl
- data/processed/val.csv
- data/processed/test.csv
- results/xgboost_design_v2_results.json

Validated outputs:

- results/release_validation/bprime_prediction_policy/metrics_by_policy.csv
- results/release_validation/bprime_prediction_policy/historical_comparison.csv
- results/release_validation/bprime_prediction_policy/run_manifest.json

Raw test Spearman: 0.670049192932404.
Clipped test Spearman: approximately 0.6700606317.

## 6. Production FoldX saved-output parser

Entry points:

- scripts/parse_foldx_ddg.py
- scripts/foldx_results.py

```bat
python scripts\parse_foldx_ddg.py --outdir results\user_runs\foldx_reparse
```

Required assets:

- results/foldx/individual_list.txt
- results/foldx/Average_tada8e_structure_Repair.fxout
- results/foldx/Dif_tada8e_structure_Repair.fxout
- results/foldx/Raw_tada8e_structure_Repair.fxout
- results/deaminase_compare/candidate_consensus_exact.csv
- data/structures/tada8e_sequence.fasta

Validated outputs:

- results/release_validation/foldx_parser_release/ddg_results.csv
- results/release_validation/foldx_parser_release/repeat_energy_checks.csv
- results/release_validation/foldx_parser_release/run_manifest.json

All 44 means and SDs matched the saved-output audit.
Twenty-three means were below zero.
SD follows the population computational-repeat convention, ddof=0.

## 7. Production protein-score consistency analysis

Entry point:

- scripts/ddg_lm_consistency.py

```bat
python scripts\ddg_lm_consistency.py --foldx results\user_runs\foldx_reparse\ddg_results.csv --outdir results\user_runs\protein_consistency
```

Also requires logprob_matrix.npy and candidate_mutations.csv in:

- results/deaminase/
- results/deaminase_650m/
- results/deaminase_if1/

Validated outputs:

- results/release_validation/protein_consistency_release/lm_ddg_merged.csv
- results/release_validation/protein_consistency_release/correlations.csv
- results/release_validation/protein_consistency_release/matrix_validation.csv
- results/release_validation/protein_consistency_release/run_manifest.json

ESM-IF1 rho=-0.4786469344608879, p=0.0010139508880828937, n=44.

Plots are optional and require matplotlib, which is not included in
the tested core environment recipe.

## 8. Development cross-checks

The following audit scripts support development verification but are not
additional mandatory steps in the normal production reanalysis workflow:

- scripts/validate_protein_saved_results.py
- scripts/validate_bprime_prediction_policy.py

Saved protein audit:

- results/release_validation/protein_saved_audit/

The B-prime audit remains the recommended entry for reproducing its
raw/clipped metric comparison.

## Reproduction levels

Distinguish:

1. Plot regeneration from saved tables.
2. Inference using an existing trained model.
3. Reanalysis of saved external-tool output.
4. Model retraining.
5. New external-tool computation.

Success at one level does not establish completion of the others.