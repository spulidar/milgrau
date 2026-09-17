"""Scientific and serialization guards for the offline candidate evidence join."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from milgrau.cli.high_column_evidence import main
from milgrau.level2.high_column_export import (
    candidate_evidence_rows, evidence_summary, write_evidence_export,
)
from milgrau.level2.rayleigh_catalogue_dataset import attach_rayleigh_candidate_catalogue
from milgrau.level2.rayleigh_candidates import catalogue_rayleigh_candidates, select_minimum_cost_accepted_candidate
from milgrau.provenance import file_sha256


def _dataset() -> xr.Dataset:
    altitude = np.arange(100.0, 2100.0, 100.0)
    molecular = np.exp(-altitude / 7000.0)
    signal = np.stack([2 * molecular, 4 * molecular])
    # A slope in the second block creates real rejected candidates.
    signal[1, 10:] *= np.linspace(1, 3, 10)
    error = np.full_like(signal, 0.1)
    time = np.asarray(["2025-11-07T16:40", "2025-11-07T17:00"], dtype="datetime64[ns]")
    config = {"inversion": {"molecular_fit": {
        "ref_alt_min_m": 300.0, "ref_alt_max_m": 1800.0, "ref_window_m": 500.0,
        "max_relative_slope": 0.2, "max_relative_variance": 0.2, "min_valid_fraction": 0.8,
    }}}
    references = []
    for block in range(2):
        candidates = catalogue_rayleigh_candidates(
            signal[block], molecular, altitude, min_altitude_m=300, max_altitude_m=1800,
            window_bins=5, max_relative_slope=0.2, max_relative_variance=0.2,
            min_valid_fraction=0.8, measured_signal_error=error[block],
        )
        references.append(select_minimum_cost_accepted_candidate(candidates).center_altitude_m)
    result = SimpleNamespace(
        block_time=time,
        signal_selection=SimpleNamespace(retrieval_input_valid_flag_block=np.ones(2, dtype=np.int8)),
        glued=SimpleNamespace(range_corrected_signal_block=signal, range_corrected_signal_error_block=error),
        molecular=SimpleNamespace(simulated_range_corrected_signal=molecular),
        rayleigh=SimpleNamespace(reference_success_flag_block=np.ones(2, dtype=np.int8),
                                reference_altitude_m_block=np.asarray(references)),
    )
    ds = xr.Dataset(
        {
            "rayleigh_reference_success_flag_block": (("block_time", "wavelength"), [[1], [1]]),
            "rayleigh_reference_altitude_m_block": (("block_time", "wavelength"), np.asarray(references)[:, None]),
            "glued_range_corrected_signal_block": (("block_time", "wavelength", "altitude"), signal[:, None]),
            "glued_range_corrected_signal_error_block": (("block_time", "wavelength", "altitude"), error[:, None]),
            "simulated_molecular_range_corrected_signal": (("wavelength", "altitude"), molecular[None]),
        }, coords={"block_time": time, "wavelength": [532], "altitude": altitude},
        attrs={"level2_product_schema_version": "3", "level2_retrieval_method_version": "4"},
    )
    return attach_rayleigh_candidate_catalogue(ds, [result], altitude, config)


def _rows(ds: xr.Dataset) -> list[dict]:
    return candidate_evidence_rows(ds, block_weights=np.array([1, 3]), effective_vertical_resolution_m=100)


def test_all_slots_exact_geometry_and_analytic_clean_window():
    ds = _dataset()
    before = ds.copy(deep=True)
    rows = _rows(ds)
    xr.testing.assert_identical(ds, before)
    assert len(rows) == 2 * ds.sizes["rayleigh_candidate"]
    assert any(not row["candidate_shape_qa_accepted"] for row in rows)
    assert sum(row["candidate_productively_selected"] for row in rows) == 2
    first = rows[0]
    window = (ds.altitude >= first["candidate_start_altitude_m"]) & (ds.altitude <= first["candidate_stop_altitude_m"])
    x = ds.simulated_molecular_range_corrected_signal.values[0, window.values]
    assert first["window_calibration_valid_bins"] == 5  # inclusive last native bin
    assert first["window_calibration_factor"] == pytest.approx(2)
    assert first["window_calibration_snr_independent"] == pytest.approx(2 * np.sqrt(np.sum(x*x)) / 0.1)
    assert first["window_calibration_snr_fully_correlated"] == pytest.approx(2 * np.sum(x*x) / (0.1 * np.sum(x)))
    assert first["subwindow_relative_disagreement"] == pytest.approx(0, abs=1e-14)
    assert first["dominant_signal_contribution_fraction"] == pytest.approx(6/7)
    assert first["temporal_candidate_persistence_fraction"] == 1
    assert first["window_contamination_fraction"] is None
    assert first["window_calibration_snr_dependence_model"] is None
    mixed = [row for row in rows if row["temporal_candidate_persistence_fraction"] == 0.25]
    assert mixed  # accepted only in first block; explicit unequal weights matter


def test_missing_error_and_unevaluated_block_stay_missing():
    ds = _dataset()
    ds["glued_range_corrected_signal_error_block"].values[:] = np.nan
    ds["rayleigh_candidate_evaluated_flag"].values[1] = 0
    ds["rayleigh_candidate_accepted_flag"].values[1] = 0
    ds["rayleigh_candidate_selected_flag"].values[1] = 0
    ds["rayleigh_candidate_unfiltered_min_cost_flag"].values[1] = 0
    ds["rayleigh_reference_success_flag_block"].values[1] = 0
    rows = _rows(ds)
    for row in rows:
        assert row["window_calibration_snr_independent"] is None
        assert row["subwindow_relative_disagreement"] is None
        assert row["dominant_signal_contribution_fraction"] is None
        assert row["temporal_candidate_persistence_fraction"] is None
        assert row["temporal_evaluated_weight_fraction"] == 0.25
    assert all(row["candidate_binwise_snr"] is None for row in rows if not row["candidate_evaluated"])


@pytest.mark.parametrize("change", ["schema", "method", "center", "stop", "weights", "resolution"])
def test_invalid_inputs_fail_before_export(change):
    ds = _dataset()
    weights, resolution = np.array([1, 3]), 100
    if change == "schema":
        ds.attrs["level2_product_schema_version"] = "2"
    elif change == "method":
        ds.attrs["level2_retrieval_method_version"] = "5"
    elif change == "center":
        ds["rayleigh_candidate_center_index"].values[0] += 1
    elif change == "stop":
        ds["rayleigh_candidate_stop_altitude_m"].values[0] += 1
    elif change == "weights":
        weights = np.array([1, np.nan])
    else:
        resolution = 0
    with pytest.raises(ValueError):
        candidate_evidence_rows(ds, block_weights=weights, effective_vertical_resolution_m=resolution)


def test_summary_serialization_and_no_overwrite(tmp_path):
    rows = _rows(_dataset())
    summary = evidence_summary(rows, np.array([0, 1000, 2000]))
    assert sum(group["count"] for group in summary["strata"]) == len(rows)
    for group in summary["strata"]:
        assert group["diagnostics"]["window_contamination_fraction"]["missing_count"] == group["count"]
    output = tmp_path / "evidence"
    write_evidence_export(output, rows, summary, {"scope": "synthetic_test"})
    exported = json.loads((output / "candidates.json").read_text())
    assert exported == rows
    with (output / "candidates.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert csv_rows[0]["window_contamination_fraction"] == ""
    with pytest.raises(FileExistsError):
        write_evidence_export(output, rows, summary, {})


def test_cli_netcdf_hash_provenance_and_mismatch(tmp_path):
    source = tmp_path / "synthetic.nc"
    _dataset().to_netcdf(source)
    output = tmp_path / "evidence"
    args = [str(source), str(output), "--block-weights", "1", "3",
            "--weight-basis", "synthetic counts", "--effective-vertical-resolution-m", "100",
            "--resolution-basis", "synthetic sampling", "--altitude-edges-m", "0", "1000", "2000",
            "--expected-sha256"]
    with pytest.raises(SystemExit):
        main(args + ["0" * 64])
    assert not output.exists()
    assert main(args + [file_sha256(source)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["provenance"]["source_level2_sha256"] == file_sha256(source)
    assert report["provenance"]["block_weights"] == [1, 3]
