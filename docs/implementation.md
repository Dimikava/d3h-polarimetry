# Implementation record

Approved design: independent `d3h-polarimetry`, two renamed models, default
special ROI/manual-POI workflow, sliding-window maps, core analysis controls,
and opt-in twist pairing. Deliver a local GitHub-ready repository, not a remote.

Completed:

- Extract the current source files without changing the parent checkout.
- Remove other physical model implementations and alternate workflows.
- Rename the package, model API, CLI flags, labels, and settings namespaces.
- Keep equations, fitting, masks, tables, and applicable output behavior.
- Add standalone packaging, a uv lockfile, CI, MIT attribution, and sample folder.
- Test real exports, TIFF loading, numerical recovery, presets, and caches.

Implementation decisions:

- Expose the existing `window_size` setting as `--window-size`, retaining 10
  pixels and stride one. No new stride behavior was introduced.
- Keep the inherited no-op `patch_data` argument/field in internal call surfaces
  so extracted call sites remain simple; the patch implementation and CLI flag
  are absent, and no inputs are rewritten.
- Preserve the source curve-fit normalization bias. Source comparisons distinguish
  extraction regressions from inherited scientific behavior.
- Use `Colormap.with_extremes` for the same bad-pixel colors, avoiding pending
  Matplotlib deprecations in the locked environment.
- Review found inherited cache gaps. Replay now renders cached twist artifacts;
  curve reuse validates input/model/angle/mask/window identity. Additional tests
  cover these behaviors. Fingerprinting allocates at most one contiguous frame.
- The local environment's default pytest temporary directory is inaccessible;
  local validation uses unique directories under ignored `.test-runs/`.

Pending external validation: supplied microscopy samples and manual desktop
selection/appearance. Linux CI is configured but has not run on GitHub locally.

Validation completed on Windows/Python 3.12:

- Full suite: 47 tests passed, including real cache reuse and twist replay.
- Reference comparison: 16 clean/noisy signal cases and both models' window maps.
- Locked independent environment and wheel/source-distribution build succeeded.
- A fresh wheel-only environment imported the installed package and ran CLI help
  with the original toolkit unavailable.
- Undefined-name static checks passed. Independent review findings were fixed
  and verified by the reviewer.

The final `uv run pytest --basetemp=.test-runs/final` run emitted one local
pytest-cache permissions warning; all 47 tests passed. A preceding full run
with `-p no:cacheprovider` passed without warnings.
