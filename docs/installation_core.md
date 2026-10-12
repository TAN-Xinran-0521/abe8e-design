# Core installation and validation

## Tested scope

The core environment supports:

- B-prime model inference.
- Current 22 unit and local-model regression tests.
- Saved HFE genome-search output reanalysis.
- B-prime raw-versus-clipped metric auditing.
- Saved FoldX output parsing.
- Protein-score correlation analysis without plots.

It does not install Cas-OFFinder, FoldX, ESM, DNABERT or the full research
training stack. It does not download hg38 or pretrained model caches.

## Tested environment

Windows; Python 3.10.20.

Direct dependency versions:

| Package | Version |
|---|---|
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.15.2 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |
| biopython | 1.87 |
| ViennaRNA | 2.7.2 |

These are pinned direct dependencies, not a complete cross-platform lock.

## Install

In Anaconda Prompt, from the repository root:

```bat
conda env create -f environments\environment-core.yml
conda activate abe8e-core-check
python -m pip check
```

Expected dependency-check output:

```text
No broken requirements found.
```

Do not install requirements-core.txt again if the environment YAML already
completed successfully; these are alternative installation descriptions.

## Test

```bat
python -m unittest discover -s tests -p "test_*.py" -v
```

Current expected result: 22 tests pass.

The local-model regression test requires:

```text
models/xgboost_design_v2.pkl
```

Only load pickle files from trusted sources.

## Examples

See the root README and docs/results_index.md for inference and reanalysis
commands and their required assets.

Run commands from the repository root unless explicitly stated otherwise.

## Optional plotting

The revised ddg_lm_consistency.py computes tables without matplotlib.

Its --plots option requires a separately installed plotting dependency.
Plotting is not part of the currently pinned and tested core recipe.

Do not install the entire protein-model stack merely to reproduce saved
score correlations.

## Evidence

- results/release_validation/clean_environment/
- results/release_validation/hfe_candidates_clean_env.csv
- results/release_validation/release_core/tests_22.txt

The new Conda environment was created on the author's existing machine.
Another-machine and independent-user validation remain pending.

## Troubleshooting

If installation or testing fails, record:

- The exact command.
- Operating system and Python version.
- Package versions.
- Complete error output.
- Whether required model and data assets are available.

Do not silently alter package versions or test expectations and then
claim exact reproduction.