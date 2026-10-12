# Release scope

## Release identity

Core ABE8e candidate design and saved-result reproduction.

This release should not be described as an unrestricted, fully validated
end-to-end implementation of every historical experiment.

## Tier A: validated and recommended

- Core HFE inference with design_sgrna.py.
- Current 22 automatic tests.
- validate_bprime_prediction_policy.py on the fixed published data.
- validate_hfe_global_ranking.py on the saved HFE outputs.
- parse_foldx_ddg.py and foldx_results.py.
- ddg_lm_consistency.py without optional plotting.
- The pinned core installation recipe on the tested Windows environment.

## Tier B: research code supplied, not fully release-validated

- Model retraining scripts.
- Generic new Cas-OFFinder execution and arbitrary-input workflows.
- SHAP, motif and some plot-generation workflows.
- ESM and ESM-IF1 inference.
- Remaining protein preparation and visualization scripts.
- DNABERT experiments and their optional dependencies.

Path correction or syntax checking alone does not validate these workflows.

## Tier C: historical or protected

- *_V0.py scripts.
- setup_deaminase_env.py, previously associated with an obsolete sequence setup.
- foldx_ddg.py: historical execution entry point remains disabled.
- Historical median-based priority_quadrant outputs.
- Old subgroup-normalized genome-hit scores.
- Earlier window-count bystander ranking.
- Original exploratory scripts with author-environment assumptions.

These are not default quick-start entry points.

## Historical evidence

Do not silently overwrite:

- Original model metrics.
- Original data splits.
- Original Cas-OFFinder outputs.
- Original FoldX individual_list, Raw, Dif and Average files.
- Earlier rounded protein results.
- Earlier test logs.

Revised results belong in separate directories with manifests.

## Remaining limitations

- Generic parser error handling is not fully covered for all formats.
- Arbitrary repeated-guide and zero-hit workflows need additional validation.
- Reference-genome search completeness has not been independently checked.
- Full research retraining and new FoldX computations have not been completed
  in the release environment.
- No external-user or cross-platform reproduction is claimed.

## Public-release gates

Before publishing:

- Confirm rights to release the original code.
- Add the chosen MIT license with the actual copyright holder.
- Review data, model and third-party redistribution permissions.
- Decide which large assets belong in versioned releases or data archives.
- Provide stable download paths and checksums for external assets.
- Check exported logs and environment files for private paths, URLs and tokens.
- Validate documented commands against the exact final asset layout.
- Remove local patch markers, caches and unnecessary third-party binaries.
- Create a real version tag and changelog entry only when the release exists.

A documented limited release is preferable to an unsupported claim that
everything can be reproduced from scratch.