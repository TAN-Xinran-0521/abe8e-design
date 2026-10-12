# True (coordinate-aware) Off-target Reranking Report

## Input
- Candidates: `D:/TZS/results/hfe_candidates.csv`
- Genome: `D:/TZS/genomes/hg38`
- Search PAM: `NNN`  |  Max mismatches: `4`
- Scenario: `therapy`  |  Device: `G`
- On-target exclusion: coordinate-based (tolerance 30 bp)

## Cas-OFFinder command
```
D:/TZS/tools/cas-offinder/cas-offinder.exe D:\TZS\results\hfe_offtarget_true\casoffinder_input.txt G D:\TZS\results\hfe_offtarget_true\casoffinder_output.txt
```

## Summary
- Candidate rows: 3
- Unique guides: 3
- Parsed hits: 2414
- On-target hits removed: 3
- True off-target hits: 2411

## Top candidates (coordinate-aware true off-target)

 rank_true_offtarget candidate_id                 grna cas_variant  predicted_efficiency  bystander_A_count  total_offtargets  offtarget_risk_norm  composite_score_true_offtarget  pareto
                   1            1 TACGTACCAGGTGGAGCACC        SpRY                 0.245                  0               654             0.906255                       -0.173877    True
                   2            2 ACGTACCAGGTGGAGCACCC      SpCas9                 0.335                  1               800             0.947571                       -0.225271    True
                   3            3 CGTACCAGGTGGAGCACCCA   SpCas9-NG                 0.288                  1               957             1.000000                       -0.259800   False

## Notes
- On-target sites excluded by real genomic coordinates from a ClinVar locus (LMNA c.1824), not by heuristic.
- `offtarget_risk_raw` = mismatch-count + position weighted aggregate of true off-target hits.
- This is genome-wide **Cas9-dependent** off-target screening only.
  Cas9-independent deaminase off-target (random ssDNA/RNA editing) is NOT captured; needs GOTI/Detect-seq.
- PAM-relaxed variants (VRQR/SpRY) trade wider targeting for potentially more off-targets.