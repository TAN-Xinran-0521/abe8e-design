# True (coordinate-aware) Off-target Reranking Report

## Input
- Candidates: `D:/TZS/results/candidates.csv`
- Genome: `D:/TZS/genomes/hg38`
- Search PAM: `NNN`  |  Max mismatches: `4`
- Scenario: `therapy`  |  Device: `G`
- On-target exclusion: coordinate-based (tolerance 30 bp)

## Cas-OFFinder command
```
D:/TZS/tools/cas-offinder/cas-offinder.exe D:\TZS\results\offtarget_true\casoffinder_input.txt G D:\TZS\results\offtarget_true\casoffinder_output.txt
```

## Summary
- Candidate rows: 3
- Unique guides: 3
- Parsed hits: 79566
- On-target hits removed: 3
- True off-target hits: 79563

## Top candidates (coordinate-aware true off-target)

 rank_true_offtarget candidate_id                 grna cas_variant  predicted_efficiency  bystander_A_count  total_offtargets  offtarget_risk_norm  composite_score_true_offtarget  pareto
                   1            2 GTCCACCCACCTGGGCTCCT        SpRY                 0.196                  0             12378             0.905492                       -0.193247    True
                   2            1 GGTCCACCCACCTGGGCTCC        VRQR                 0.218                  0             52469             1.000000                       -0.212800    True
                   3            3 TCCACCCACCTGGGCTCCTG   SpCas9-NG                 0.261                  1             14716             0.933956                       -0.250787    True

## Notes
- On-target sites excluded by real genomic coordinates from a ClinVar locus (LMNA c.1824), not by heuristic.
- `offtarget_risk_raw` = mismatch-count + position weighted aggregate of true off-target hits.
- This is genome-wide **Cas9-dependent** off-target screening only.
  Cas9-independent deaminase off-target (random ssDNA/RNA editing) is NOT captured; needs GOTI/Detect-seq.
- PAM-relaxed variants (VRQR/SpRY) trade wider targeting for potentially more off-targets.