# Extraction provenance

This project was extracted on 2026-11-12 from the current working files of
Dimikava's `tiff-analyzer`, licensed under MIT. It includes existing local changes
present in that checkout; it is not asserted to match a published release.
The original copyright and license are retained in `LICENSE`.

The source repository was left unchanged. Only dependencies of the D3h special
ROI/POI workflow were retained. Model names changed as follows:

| Source | Standalone |
|---|---|
| d3hanal / SHGd3hModel | d3hfixed / D3hFixedModel |
| d3hmov / SHGCos2MovModel | d3hrotating / D3hRotatingModel |
| FFTMode.SHGd3hanal | FFTMode.D3HFIXED |
| FFTMode.SHGd3hanalMOV | FFTMode.D3HROTATING |

The extraction removes unrelated model branches, vendor-reader plugins, template
composition, alternate workflows, and frame-patching execution. The original
reader/axis contracts, fitting, plotting, intensity-mask growth, special workflow,
cache handling, and export code remain local to `d3h_polarimetry`.

The CLI now defaults to the special workflow and exposes `--window-size` directly
(default 10, stride one). New metadata records the renamed model. The package and
settings directories use their own namespace. No runtime import from the original
toolkit is required.

Two focused cache fixes accompany extraction: special-workflow replay now
regenerates saved twist maps/histograms, and curve-cache reuse validates identity
before applying fitted maps. Sidecar metadata adds the input fingerprint, angular
coordinates, and A0 mask settings; incompatible or metadata-free caches require a
fresh analysis. These changes do not alter model equations or fitted values.

Before extraction completion, deterministic reference comparisons ran on both
models at 0°, 12°, 37°, and 59°, with zero noise and seeded Gaussian noise of
standard deviation 120. FFT/fit result keys, scalars, and prediction arrays matched
the source within 1e-12. Both models' FFT/fit window maps, A0 maps/masks, and window
statistics were compared as well. The retained regression tests additionally
check analytical coefficients, noisy recovery, twist arithmetic, and exports.

`docs/source_checksums.json` records SHA-256 hashes of the source modules at
extraction time. To repeat the numerical comparison with a separate checkout:

```powershell
uv run python scripts/compare_reference.py /path/to/tiff-analyzer
```

Real sample-data parity and manual desktop interaction remain to be checked.
