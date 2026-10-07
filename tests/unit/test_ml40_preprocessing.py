"""Unit tests for ml40 preprocessing.detect_system."""
import pytest

from app.domain.services.exceptions import UnknownDiagnosisSystemError
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.constants import (
    CYCLE_COLUMNS,
    RAW_INPUT_COLUMNS,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.preprocessing import detect_system


@pytest.mark.parametrize("system", ["refrigeracion", "aireado"])
def test_detect_system_single_subsystem(system):
    assert detect_system(CYCLE_COLUMNS + RAW_INPUT_COLUMNS[system]) == system


def test_detect_system_rejects_mixed_subsystems():
    mixed = CYCLE_COLUMNS + RAW_INPUT_COLUMNS["refrigeracion"] + RAW_INPUT_COLUMNS["aireado"]
    with pytest.raises(UnknownDiagnosisSystemError, match="ambos subsistemas"):
        detect_system(mixed)


@pytest.mark.parametrize("system", ["refrigeracion", "aireado"])
def test_detect_system_explicit_matching_columns(system):
    assert detect_system(CYCLE_COLUMNS + RAW_INPUT_COLUMNS[system], system) == system


def test_detect_system_explicit_rejects_other_subsystem_csv():
    with pytest.raises(UnknownDiagnosisSystemError, match="Has seleccionado el subsistema 'refrigeracion'"):
        detect_system(CYCLE_COLUMNS + RAW_INPUT_COLUMNS["aireado"], "refrigeracion")


def test_detect_system_explicit_rejects_mixed_csv():
    mixed = CYCLE_COLUMNS + RAW_INPUT_COLUMNS["refrigeracion"] + RAW_INPUT_COLUMNS["aireado"]
    with pytest.raises(UnknownDiagnosisSystemError, match="ambos subsistemas"):
        detect_system(mixed, "aireado")


def test_detect_system_explicit_without_signature_defers_to_column_check():
    # No signature columns at all: returns the chosen system so validate_raw_input can report
    # exactly which sensors are missing.
    assert detect_system(CYCLE_COLUMNS + ["T_amb"], "aireado") == "aireado"


def test_detect_system_rejects_unknown_explicit_system():
    with pytest.raises(UnknownDiagnosisSystemError, match="no reconocido"):
        detect_system(CYCLE_COLUMNS + RAW_INPUT_COLUMNS["aireado"], "otro")
