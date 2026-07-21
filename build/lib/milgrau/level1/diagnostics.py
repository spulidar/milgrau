"""Level 1 correction diagnostics and metadata assembly."""

from __future__ import annotations

import numpy as np
import xarray as xr


def finalize_correction_dataset(
    final_ds: xr.Dataset,
    status_records: list[tuple[str, int, int]],
    diagnostic_records: list[dict[str, object]],
) -> xr.Dataset:
    """Attach correction status and diagnostics to the concatenated Level 1 dataset."""
    final_ds["corrected_signal"].attrs.update(
        {
            "long_name": "Instrumentally corrected lidar signal",
            "description": "Signal after dark-current, dead-time, bin-shift and background corrections before range correction",
            "units": "channel native corrected units",
        }
    )
    final_ds["corrected_signal_error"].attrs.update(
        {"long_name": "One-sigma uncertainty of instrumentally corrected lidar signal", "units": "channel native corrected units"}
    )
    final_ds["range_corrected_signal"].attrs.update(
        {
            "long_name": "Range Corrected Signal",
            "description": "Instrumentally corrected signal multiplied by range squared",
            "units": "a.u. m^2",
        }
    )
    final_ds["range_corrected_signal_error"].attrs.update({"long_name": "One-sigma uncertainty of Range Corrected Signal", "units": "a.u. m^2"})
    final_ds["pc_saturation_mask"].attrs.update(
        {
            "long_name": "Photon-counting saturation/dead-time clipping mask",
            "description": "1 where the photon-counting signal was flagged as saturated or dead-time clipped after bin-shift alignment; 0 elsewhere. Analog channels are always 0.",
            "flag_values": "0, 1",
            "flag_meanings": "valid saturated_or_clipped",
        }
    )

    status_map = {name: (ok, dc) for name, ok, dc in status_records}
    final_channels = final_ds.channel.values.astype(str)
    final_ds["channel_correction_success"] = xr.DataArray(
        [status_map.get(ch, (0, 0))[0] for ch in final_channels],
        dims=["channel"],
        coords={"channel": final_channels},
    ).astype(np.int8)
    final_ds["dark_current_used"] = xr.DataArray(
        [status_map.get(ch, (0, 0))[1] for ch in final_channels],
        dims=["channel"],
        coords={"channel": final_channels},
    ).astype(np.int8)
    final_ds["channel_correction_success"].attrs.update({"flag_values": "0, 1", "flag_meanings": "failed success"})
    final_ds["dark_current_used"].attrs.update({"flag_values": "0, 1", "flag_meanings": "not_used used"})

    diag_by_channel = {str(record["channel"]): record for record in diagnostic_records}
    final_ds["deadtime_correction_applied"] = xr.DataArray(
        [diag_by_channel[ch]["deadtime_correction_applied"] for ch in final_channels],
        dims=["channel"],
        coords={"channel": final_channels},
    ).astype(np.int8)
    final_ds["deadtime_min_denominator_observed"] = xr.DataArray(
        [diag_by_channel[ch]["deadtime_min_denominator_observed"] for ch in final_channels],
        dims=["channel"],
        coords={"channel": final_channels},
    ).astype(np.float32)
    final_ds["deadtime_min_denominator_allowed"] = xr.DataArray(
        [diag_by_channel[ch]["deadtime_min_denominator_allowed"] for ch in final_channels],
        dims=["channel"],
        coords={"channel": final_channels},
    ).astype(np.float32)
    final_ds["bin_shift_bins"] = xr.DataArray(
        [diag_by_channel[ch]["bin_shift_bins"] for ch in final_channels],
        dims=["channel"],
        coords={"channel": final_channels},
    ).astype(np.int16)

    deadtime_fraction = np.stack([np.asarray(diag_by_channel[ch]["deadtime_clipping_fraction"].values) for ch in final_channels], axis=1)
    pc_saturation_fraction = np.stack([np.asarray(diag_by_channel[ch]["pc_saturation_fraction"].values) for ch in final_channels], axis=1)
    bin_shift_fraction = np.stack([np.asarray(diag_by_channel[ch]["bin_shift_invalid_fraction"].values) for ch in final_channels], axis=1)
    final_ds["deadtime_clipping_fraction"] = xr.DataArray(
        deadtime_fraction,
        dims=["time", "channel"],
        coords={"time": final_ds.time, "channel": final_channels},
    ).astype(np.float32)
    final_ds["pc_saturation_fraction"] = xr.DataArray(
        pc_saturation_fraction,
        dims=["time", "channel"],
        coords={"time": final_ds.time, "channel": final_channels},
    ).astype(np.float32)
    final_ds["bin_shift_invalid_fraction"] = xr.DataArray(
        bin_shift_fraction,
        dims=["time", "channel"],
        coords={"time": final_ds.time, "channel": final_channels},
    ).astype(np.float32)
    final_ds["deadtime_correction_applied"].attrs.update({"flag_values": "0, 1", "flag_meanings": "not_applied applied"})
    final_ds["deadtime_clipping_fraction"].attrs.update({"units": "1", "description": "Fraction of altitude bins where the non-paralyzable dead-time denominator was clipped."})
    final_ds["pc_saturation_fraction"].attrs.update({"units": "1", "description": "Fraction of altitude bins where pc_saturation_mask equals 1."})
    final_ds["bin_shift_invalid_fraction"].attrs.update({"units": "1", "description": "Fraction of altitude bins introduced by bin-shift alignment and marked as NaN."})
    final_ds["bin_shift_bins"].attrs.update({"units": "bins"})
    return final_ds
