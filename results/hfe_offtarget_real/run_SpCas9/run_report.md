# True (coordinate-aware) Off-target Reranking Report

## Input
- Candidates: `D:\TZS\results\hfe_offtarget_real\cand_SpCas9.csv`
- Genome: `D:/TZS/genomes/hg38`
- Search PAM: `NGG`  |  Max mismatches: `4`
- Scenario: `therapy`  |  Device: `G`
- On-target exclusion: coordinate-based (tolerance 30 bp)

## Cas-OFFinder command
```
D:\TZS\tools\cas-offinder\cas-offinder.exe D:\TZS\results\hfe_offtarget_real\run_SpCas9\casoffinder_input.txt G D:\TZS\results\hfe_offtarget_real\run_SpCas9\casoffinder_output.txt
```

## Summary
- Candidate rows: 1
- Unique guides: 1
- Parsed hits: 80
- On-target hits removed: 1
- True off-target hits: 79

## Top candidates (coordinate-aware true off-target)

 rank_true_offtarget candidate_id                 grna cas_variant  predicted_efficiency  bystander_A_count  total_offtargets  offtarget_risk_norm  composite_score_true_offtarget  pareto
                   1            2 ACGTACCAGGTGGAGCACCC      SpCas9                 0.335                  1                79                  1.0                          -0.241    True

## Notes
- On-target sites excluded by real genomic coordinates from a ClinVar locus (LMNA c.1824), not by heuristic.
- `offtarget_risk_raw` = mismatch-count + position weighted aggregate of true off-target hits.
- This is genome-wide **Cas9-dependent** off-target screening only.
  Cas9-independent deaminase off-target (random ssDNA/RNA editing) is NOT captured; needs GOTI/Detect-seq.
- PAM-relaxed variants (VRQR/SpRY) trade wider targeting for potentially more off-targets.