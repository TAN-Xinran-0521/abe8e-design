# Third-party resources and notices

## Status

This is a provenance and rights-review checklist, not a declaration that
all resources may be redistributed.

The project's original code is intended for MIT licensing.
That choice does not automatically license third-party code, data,
pretrained weights, tool resources or derived assets.

Do not assign an unverified license to a resource.

## Research resources

| Resource | Use | Known local identity | Distribution status |
|---|---|---|---|
| BE-dataHIVE | Training and interpretation data | ABE8e efficiency/bystander snapshots | Permission and citation review pending |
| hg38 | Genome-search reference | Original hg38 working files | Do not include full genome in main repository |
| ClinVar | Example variant provenance | HFE and LMNA examples | Document source records and conditions |
| Addgene | TadA-8e sequence provenance | Plasmid 138489 sequence material | Redistribution review pending |
| ESM2 | Protein sequence scoring | esm2_t12_35M_UR50D | Do not copy pretrained cache by default |
| ESM1v | Protein sequence scoring | esm1v_t33_650M_UR90S_1 | Do not copy pretrained cache by default |
| ESM-IF1 | Structure-conditioned scoring | esm_if1_gvp4_t16_142M_UR50 | Do not copy pretrained cache by default |
| ESMFold | Predicted structure | Recorded ESMFold-derived PDB | Document generation provenance and terms |
| Cas-OFFinder | Potential genome-match search | Recorded v2.4.1 workflow | External installation; review terms |
| FoldX | Predicted stability changes | Saved output identifies FoldX 5.1 | Do not distribute program or restricted resources |
| DNABERT-2 | Exploratory sequence representation | DNABERT-2-117M | Exploratory; no pretrained cache bundled |

## Python dependencies

The core environment depends on independently maintained packages:

- NumPy
- pandas
- SciPy
- scikit-learn
- XGBoost
- Biopython
- ViennaRNA

These packages retain their own licenses and notices.
The project does not claim their code as original work.

Additional research workflows use optional third-party dependencies.
Document those before advertising the corresponding workflow as supported.

## Files not to publish without explicit review

- FoldX executable and installation archives.
- FoldX rotabase.txt and molecules resources.
- Cas-OFFinder executable archives, unless redistribution is confirmed.
- Hugging Face and Torch pretrained cache directories.
- Addgene source materials with unconfirmed redistribution conditions.
- Raw/processed database snapshots without confirmed redistribution rights.
- Private access tokens, local credentials and authenticated URLs.

## Release checklist for each external asset

Record:

1. Official source URL.
2. Exact version or identifier.
3. Acquisition date where relevant.
4. Applicable license/terms.
5. Whether redistribution is permitted.
6. Required attribution/citation.
7. File checksum.
8. If not redistributed, a legal acquisition and reconstruction path.

The presence of a file in a publicly accessible source does not by itself
establish permission to repackage it under MIT.

## Derived outputs

Small score matrices, candidate tables, predicted structures and tool
outputs should have documented provenance.

Do not assume either that all derived outputs are prohibited or that all
are unrestricted. Review the applicable source/tool terms.