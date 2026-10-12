# Data and resource provenance

This document records known provenance and explicitly identifies missing
release metadata. Public availability does not automatically permit
redistribution.

## 1. BE-dataHIVE

Use:

- ABE8e efficiency modelling.
- Position-attribution and motif analyses.
- Historical cross-study diagnostics.

Local snapshots:

- data/raw/efficiency_ABE8e_raw.csv
- data/raw/bystander_ABE8e_raw.csv

Historical source size: 28962 ABE8e efficiency rows.

Core preprocessing:

- study_id=3.
- efficiency_full_grna_reported as label.
- Remove rows lacking the target label.
- Remove rows lacking energy_1 through energy_24.
- Group split using exact full_context_sequence values.
- Fixed outputs: 15530 train, 3328 validation, 3329 test.

Study 4 is a different-label external diagnostic dataset and must not be
silently pooled with the same-source formal test set.

Before public distribution, record:

- Official source URL/API endpoint.
- Actual snapshot acquisition date.
- Dataset version if available.
- Original snapshot SHA256.
- Applicable terms and redistribution permission.
- Required citation.

Do not infer these missing details from local file modification times.

## 2. Fixed processed splits

Local files:

- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv
- data/processed/external_test_study4.csv

The original random split depends on the input sequence ordering as well
as the random seed. Preserve the original snapshot and fixed assignments.

The patched preprocessing script can write a split manifest, but the
presence of its code does not prove that the manifest has already been
generated and validated.

Do not rerun preprocessing merely to fill a documentation gap if it would
overwrite the historical split files.

Some processed-file hashes are recorded in the B-prime audit manifest.
A complete public data-release manifest remains to be finalized.

## 3. HFE and LMNA examples

Local directory:

```text
data/clinvar/
```

Examples:

- HFE c.845G>A.
- LMNA c.1824C>T.

The patient_context files are simulated variant contexts constructed from
public variant information and a reference sequence. They are not actual
patient sequencing records.

Release metadata should include:

- Variant identifier and reference transcript where applicable.
- GRCh38 coordinate convention.
- Reference and alternate alleles.
- Source record and access date.
- Context-generation procedure.

## 4. hg38 reference genome

Used for saved Cas-OFFinder searches.

The full genome is not intended to be included in the main repository.

Required reproducibility metadata:

- Download source.
- Exact assembly/file identity.
- Included contigs.
- File SHA256.
- Whether alternate or decoy sequences are included.

The original input path alone is not sufficient provenance.
The reference-genome checksum has not yet been independently verified
in the recorded saved-output reanalysis.

## 5. TadA-8e sequence and predicted structure

Local directory:

```text
data/structures/
```

Recorded sources:

- Addgene plasmid 138489 sequence material.
- Extracted 166-residue TadA-8e reference.
- ESMFold-predicted structure.

Before public release:

- Confirm Addgene material redistribution conditions.
- Record exact source and citation.
- Record sequence/structure generation provenance.
- Keep sequence numbering distinct from literature numbering.
- Do not assume that a universal +1 mapping applies to every reference.

The saved-output parser validates mutations against the supplied FASTA.
It does not replace explicit PDB/reference verification before a new
FoldX calculation.

## 6. Protein-model scoring assets

Saved result directories:

- results/deaminase/
- results/deaminase_650m/
- results/deaminase_if1/

Each includes small derived score matrices and candidate tables.

Matrices used in validation:

- Shape: 166 x 20.
- Column order: ACDEFGHIKLMNPQRSTVWY.
- Values: finite float32 arrays.

These are derived scoring outputs, not the large pretrained model weights.
Their provenance and permitted distribution still need documentation.

## 7. Saved external-tool outputs

Cas-OFFinder:

- Preserve input, command and raw output.
- Document PAM rule, mismatch limit and target exclusion convention.

FoldX:

- Preserve individual_list.txt, Average, Dif and Raw output.
- Do not distribute the program or restricted resource files merely
  because they appeared inside the working directory.

Output redistribution conditions should be checked separately from
software redistribution conditions.

## 8. Public asset packaging

If data or models are placed outside ordinary Git history, provide:

- Versioned asset URL.
- Filename.
- Size.
- SHA256.
- Compatible software release.
- License or access conditions.
- Download and placement instructions.

Do not publish temporary or inaccessible asset links as the sole
reproduction route.