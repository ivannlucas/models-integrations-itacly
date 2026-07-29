"""Vendored copy of the ml46 (DNSL) training/inference pipeline.

Adapted from the AI team's delivered code (inbox/a46/codigo/) with import paths
rewritten to this package and the YAML-config/repo-path machinery stripped out
(the plugin uses ArtifactStore + explicit dicts instead). The algorithmic logic
(feature engineering, windowing, model architecture, alert policy) is kept as
close to verbatim as possible so predictions stay numerically reproducible
against the golden dataset in inbox/a46/manifest.yaml.
"""
