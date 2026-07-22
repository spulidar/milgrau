# Incremental processing and provenance

MILGRAU reuses a generated artifact only when its output, inputs, relevant
configuration, software identity, and product variant still match. Merely finding
the output path is never sufficient.

## Manifest contract

Every reusable artifact has a sidecar named
`<artifact suffix>.provenance.json`. The sidecar contains:

- provenance format version;
- product kind;
- SHA-256 fingerprint of the canonical provenance payload;
- input file names, sizes, and SHA-256 digests;
- the code-audited relevant configuration subset;
- installed MILGRAU version and SHA-256 digest of all `milgrau/**/*.py` sources;
- product-specific variants;
- generated artifact name, size, and SHA-256 digest.

JSON keys and input signatures are sorted before hashing. Absolute paths,
processing time, filesystem modification time, log configuration, output
directories, worker counts, and the `processing.incremental` switch are not part
of the fingerprint.

Manifests are written atomically only after the artifact is complete. NetCDF
reuse additionally requires opening and fully loading the file and satisfying
the structural contract for its level. Visual artifacts must be non-empty and
must match the output digest recorded in their sidecar.

An existing artifact without a valid version-1 sidecar is treated as stale and
is regenerated once. A missing, empty, altered, unreadable, or contract-invalid
artifact is also stale even when a sidecar exists.

## Invalidation policy by product

| Product | Input identity | Relevant configuration | Variant and integrity |
| --- | --- | --- | --- |
| Level 0 / LIBIDS | Content of every measurement and associated dark-current file; inventory source role and association metadata | Site/location timezone; laser-shot tolerance; dark-current association window; Level 0 physics fields; hardware channel IDs | Measurement group ID; Level 0 NetCDF contract |
| Level 1 / LIPANCORA | Level 0 NetCDF content | Speed of light aliases; channel corrections; active PBL controls; radiosonde station ID aliases | Level 1 NetCDF contract |
| Level 2 / LEBEAR | Level 1 NetCDF content | Station altitude aliases and every active inversion setting | UTC start, stop, and output tag; Level 2 NetCDF contract |
| LIRACOS quicklook | Level 1 NetCDF and logo assets that exist at render time | Output format, DPI, and active quicklook rendering controls | Exact channel and maximum altitude; image digest |
| LIRACOS global mean | Level 1 NetCDF and logo assets that exist at render time | Output format, DPI, configured channels, and altitude ranges | Channels present and plotted; image digest |
| LEBEAR Level 2 QA | Level 1 and Level 2 NetCDFs plus logo assets that exist at render time | Output format, DPI, and active Level 2 QA controls | Indexed set of generated plots; every plot has its own digest and sidecar |

Changing another selected channel or altitude does not invalidate an already
rendered individual quicklook because its exact channel and altitude are in that
plot's variant. Those selections do invalidate the combined global-mean plot.
Changing Level 2 QA settings does not invalidate the scientific Level 2 NetCDF;
it invalidates only the QA set.

## Engineering/science boundary

The configuration subsets are an inventory of values consumed by the current
code, not a scientific judgment that a parameter is unimportant. All active
scientific controls consumed by a product are included conservatively. The only
scientific-looking settings excluded are controls already classified as dormant
by ENG-017 and not connected to the production path:

- Level 2 orchestration `enabled` and `interactive_qa`;
- selective `inversion.products` writing;
- experimental `inversion.cloud_screening`;
- quicklook `show_pbl`, `show_tropopause`, and `mean_profile_smooth_bins`;
- Level 2 QA `max_altitude_km`.

If any dormant control becomes operational, the implementing task must add it
to the corresponding provenance subset and tests in the same change.

Cache directory paths are operational and are excluded. Surface-weather and
radiosonde values actually used are embedded in the generated NetCDFs, but a
later change to a remote provider or cached contextual payload alone is not
currently detected as an invalidator. Reprocessing such contextual data must be
requested explicitly by disabling incremental mode for that run. Connecting
external-payload digests to the fingerprint remains a follow-up provenance
extension; this limitation does not permit silently activating fallback or
provider controls that ENG-017 classified as dormant.
