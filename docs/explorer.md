# MILGRAU Explorer — inspecting processed observations

MILGRAU Explorer is a **local, read-only scientific viewer** for saved lidar processing results. It can help inspect signals, atmosphere, quality diagnostics and aerosol optical products without modifying source NetCDF files or rerunning retrievals.

## Start Explorer

Explorer is included in the normal installation:

~~~bash
python -m pip install -e .
milgrau-explorer
~~~

In your browser open **http://127.0.0.1:8765**. If the project configuration is elsewhere:

~~~bash
milgrau-explorer --config /path/to/config.yaml
~~~

The viewer uses the configured processed-data location. A session not shown may lack the relevant scientific NetCDF product or may be outside the configured processing directory.

## Select a scientific observation

Choose a continuous session from the session selector. A canonical Level 1 is used for corrected-signal inspection; the viewer does not silently select another derived file when Level 1 is missing.

A session can contain more than one solar regime/segment. Check the observation timestamps, channel availability and UTC ranges before comparing visualizations.

## Main scientific views

| View | What to look for |
| --- | --- |
| **Overview** | Session period, available Level 0/1/2 products, channels, segment sequence and figures |
| **RCS / Signals** | Time–height heatmap and mean profile of the selected saved Level 1 signal |
| **Atmospheric Profiles** | Level 1 temperature/pressure at selected atmosphere times and recorded source |
| **L2 Products** | Saved aerosol backscatter/extinction or temporal support by wavelength |
| **Scientific QA** | Saved background, detector and retrieval/gluing diagnostics |
| **Metadata** | Dimensions, units, variables, source information and provenance |

Some tabs or variables may be absent if the corresponding scientific product was not generated.

## Reading a corrected-signal figure

The heatmap represents a saved Level 1 signal such as `range_corrected_signal`, `corrected_signal` or `signal_pre_background`. These are **not interchangeable**. Confirm the selected variable and axis units before comparing sessions.

The accompanying vertical profile summarizes the selected signal over time. Missing observation periods appear as gaps rather than being treated as zeros.

You can zoom, pan, change palette, adjust the colorbar range and export an image. Display-time cell aggregation and robust-percentile color scaling help visualize large files but **do not change the stored scientific samples**.

A logarithmic display suppresses nonpositive values in the plot; it does not make those values invalid or remove them from the NetCDF.

## Reading atmospheric and Level 2 views

The atmosphere plot shows the Level 1 source actually saved, not a new meteorological request. It is important to inspect per-hour source and fallback information: the same observation session can contain ERA5, Wyoming/IGRA2 and USSA76 at different times or heights.

For Level 2, view aerosol optical products together with valid support and uncertainty information. A colored pixel in a saved diagnostic field does not itself certify an accepted aerosol inversion. Review the molecular reference, lidar ratio and branch endpoints in the underlying product.

## Practical limitations

- Explorer does not carry out scientific processing, recalibration or data correction.
- Editing colors, axes and plots is a **view change** only.
- Heatmap display aggregates cells for responsiveness; export of a view is not a replacement for a scientific NetCDF.
- The viewer is not evidence that missing data, an optical retrieval or a remote SCC result has been validated.
- For analysis or publication, cite the actual software version and NetCDF source product, not only a screenshot.

For a concise command-line overview instead of a browser, run:

~~~bash
milgrau-inspect SESSION_ID
~~~

Learn more in [LIPANCORA](lipancora.md), [LEBEAR](lebear.md), [Atmospheric profiles](atmospheric_profiles.md) and [Scientific quality](scientific_quality.md).
