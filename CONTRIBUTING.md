# Contributing

Thank you for helping improve the reproducibility and usability of this
research software.

## Start with the validated scope

Read:

- README.md
- docs/release_scope.md
- docs/installation_core.md
- docs/results_index.md

Some historical scripts are supplied for research provenance but are not
recommended production entry points.

## Development environment

From Anaconda Prompt at the repository root:

```bat
conda env create -f environments\environment-core.yml
conda activate abe8e-core-check
python -m pip check
```

Run tests:

```bat
python -m unittest discover -s tests -p "test_*.py" -v
```

The current expected total is 22 tests.

## Report a problem

Include:

- Commit or release version.
- Operating system.
- Python and relevant package versions.
- Exact command.
- Minimal non-sensitive input.
- Complete error output.
- Expected versus observed behavior.

Do not upload patient personal information, credentials, private datasets
or unlicensed third-party resources.

## Code changes

- Keep training, inference and calibration feature recipes consistent.
- Preserve explicit coordinate conventions.
- Do not silently drop malformed rows or replace missing results with zero risk.
- Use full precision for calculations; round only for presentation.
- Add tests for changes to parsing, scoring, coordinate handling and feature order.
- Keep historical results separate from revised outputs.
- Do not modify expected test values merely to make a failed test pass.
- Record scientifically meaningful result changes in CHANGELOG.md.

## Scientific statements

Distinguish:

- Prediction from measurement.
- Genome matches from experimentally observed off-target editing.
- Model nomination from proven functional improvement.
- Saved-output reanalysis from a new external-tool run.
- Author-machine validation from independent-user reproduction.

## Data and artifacts

Do not commit:

- Python caches.
- Full pretrained-model caches.
- Full reference genomes without an approved distribution plan.
- FoldX software or restricted resources.
- Large duplicate intermediate files.
- Personal paths or authenticated download URLs.

Document source, version, license and checksum for new assets.

## Pull requests

Describe:

1. The problem addressed.
2. Files changed.
3. Tests executed and their results.
4. Whether numerical outputs change.
5. Whether documentation or licensing is affected.

Before submitting, confirm that contributions can be released under the
project's actual published license. The intended original-code license
is MIT; do not treat this statement as a substitute for the LICENSE file.