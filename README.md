# d3h-polarimetry

Standalone D3h SHG polarization analysis, extracted from Dimikava's MIT-licensed
`tiff-analyzer`. Supports **d3hfixed** (formerly d3hanal) and **d3hrotating**
(formerly d3hmov). The default is the special ROI/manual-POI workflow, including
whole-image sliding-window analysis. No installation of the original toolkit is needed.

## Install and run

Requires Python 3.10+ and a desktop with Tk support for interactive selection.
On Linux, install your distribution's `python3-tk` package if necessary.

```powershell
uv sync --locked
uv run d3h-polarization --help
uv run d3h-polarization --images "sample_data/experiment" --d3hfixed
uv run d3h-polarization --images "sample_data/experiment" --d3hrotating --twist-angle
```

Alternatively, `python -m pip install .` installs the command. From a checkout,
`uv run python run_polarization.py` opens the input folder picker.

## Workflow

1. Supply a TIFF file or folder of ordered TIFF frames. Confirm the polarization
   axis, angle origin, angle step, and units in the axis dialog. Frame count alone
   does not establish angular sampling. Use `--axis-mapping` and `--no-axis-gui`
   for a previously saved explicit mapping.
2. Select a crop (or keep the full image), inspect the intensity mask, and draw ROIs.
3. Select manual POIs within each ROI. ROIs are named ROI-A, ROI-B, etc.; POIs
   are numbered globally. With `--twist-angle`, select pairs within each ROI.
4. The program computes FFT and curve-fit results, whole-image window maps,
   ROI maps, POI polar plots, histograms, overlays, tables, and an analysis cache.

Useful options include `--fft-only`, `--optimizer`, `--window-size`,
`--r2-threshold`, `--a0-mask-percentile`, `--dat-file`, `--save-dir`,
`--figure-format png|svg|mixed`, and histogram/map font controls. Sliding windows
advance by one pixel, as in the original. `--roimpois-special` is accepted but
unnecessary. Twist analysis is disabled unless `--twist-angle` is supplied.

The default R² threshold is 0.8, the A0 intensity percentile is 50, and the
window size is 10 pixels. TIFF folders load up to 180 frames by default;
use `--num-frames` to change this. Single TIFF files use their resolved angle axis.
An optional DAT file provides scale calibration; analysis can continue without it.

Results follow the original folder layout:
`~/Desktop/Results/<experiment>/ROIs/roimpois_special/`. An explicit
`--save-dir` is also followed by the `roimpois_special` subfolder.
Use separate save directories for different models to avoid overwriting results.

```powershell
uv run d3h-polarization --images "sample_data/experiment" --fft-only --figure-format svg
uv run d3h-polarization --images "sample_data/experiment" --replay-cache
uv run d3h-polarization --images "sample_data/experiment" --reuse-curve-cache auto
uv run d3h-polarization --images "sample_data/experiment" --save-preset my-settings
uv run d3h-polarization --images "sample_data/experiment" --preset my-settings
```

Replay redraws saved results without refitting. Curve-cache reuse retains interactive
selection and reuses saved fit parameters; its crop identity checks are preserved.
Reusable caches are checked against the dataset, pixel fingerprint, angular
coordinates, model, window size, and intensity-mask settings. Incompatible caches
are rejected. `--ignore-cache-crop` explicitly bypasses crop/shape, cropped-pixel
fingerprint, and derived intensity-cutoff checks; the remaining identity checks
still apply.
Only this project's caches are supported. See `--help` for all controls.

## Models and interpretation

All numerical angles are radians; displayed/exported orientations are degrees.

| Model | Intensity equation | Legacy FFT bin |
|---|---|---|
| d3hfixed | `B + A*cos(3*phi - 2*alpha)^2` | 4 |
| d3hrotating | `B + A*cos(3*phi - 3*alpha)^2` | 6 |

The extraction preserves the original FFT bin and phase recovery algorithm,
optimizer starts/fallbacks, 60° orientation convention, filtering, and twist
arithmetic. These bin indices assume the original full-turn sampling convention;
explicitly defining an angle axis does not make legacy FFT recovery valid for
arbitrary scan spans or irregular grids. No new angular resampling is introduced.

Twist values use `abs(2*(POI2_mean - orientation)) % 60`, with propagated
`sigma = 2*sqrt(std1^2 + std2^2)`. Preserve pair ordering when comparing runs.
The temporary `--temp` histogram filter and `--temp-fix` ROI-A adjustments are
available but disabled by default; they modify displayed/derived values and
should be recorded if used. See [analysis details](docs/analysis.md).

Other models, special2, standard/regular ROI workflows, simulation, stitching,
registration, batch manifests, template composition, and source-file patching
are excluded.

## Sample data and development

Place local data under [sample_data](sample_data/README.md). Images, metadata,
results, and caches are excluded from Git. Real-data comparison awaits supplied samples.

```powershell
uv sync --locked
uv run pytest
```

The source retains the MIT license and original attribution. See
[provenance](docs/provenance.md) for extraction and validation details.
