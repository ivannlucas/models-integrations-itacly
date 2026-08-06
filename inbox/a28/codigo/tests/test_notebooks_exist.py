from __future__ import annotations

from tests.conftest import REPO_ROOT


EXPECTED_NOTEBOOKS = [
    "00_data_sources_audit.ipynb",
    "01_raw_data_profile.ipynb",
    "02_external_context_eda.ipynb",
    "03_synthetic_plant_layer_eda.ipynb",
    "04_feature_engineering_audit.ipynb",
    "05_modeling_dataset_eda.ipynb",
    "06_split_validation_and_leakage_audit.ipynb",
    "07_training_and_policy_results_eda.ipynb",
]


def test_expected_notebooks_and_runner_exist() -> None:
    notebook_dir = REPO_ROOT / "notebooks"
    for notebook_name in EXPECTED_NOTEBOOKS:
        assert (notebook_dir / notebook_name).exists(), notebook_name
    assert (REPO_ROOT / "scripts" / "run_notebooks.py").exists()
