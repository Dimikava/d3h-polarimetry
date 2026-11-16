# Working on d3h-polarimetry

Read `README.md`, `docs/analysis.md`, and `docs/provenance.md` before editing.
This is an independent repository; never import from the original tiff-analyzer.

Keep changes focused. Preserve equations, radians/degrees conventions, harmonic
bins, fitting behavior, quality gates, mask logic, cache structure, table fields,
and output routing unless the user explicitly requests a change. Check current
code and focused tests before editing. Do not overwrite unrelated user changes.

Runtime code is under `src/d3h_polarimetry/`. The special workflow is owned by
`pipeline_workflow_special.py`; numerical SHG fitting is in `polarization_shg.py`;
sliding windows and twist arithmetic are in `pipeline_sliding.py` and
`pipeline_arithmetic.py`. Keep the `pipeline.py` facade small.

Run `uv run pytest`, `uv run d3h-polarization --help`, and `git diff --check`.
For changes to science, compare before/after numerical outputs explicitly.
The user-supplied TIFF images and DAT files under sample_data/ are intentionally
versioned. Do not commit other microscopy data, result tables, caches,
environments, or build artifacts.
