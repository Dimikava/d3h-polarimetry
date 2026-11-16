# Analysis conventions

The data boundary is a NumPy cube `(polarization_angle, y, x)`, with angles in
radians. TIFF decoding is strict: sorted folder members must have consistent
shapes/types; corrupt members fail the read. Multi-series TIFFs require an
explicit series. The default decoded size limit is 8 GiB. Reader and axis
provenance are written beside analysis results.

## Preserved numerical behavior

- `d3hfixed`: `B + A*cos(3*phi - 2*alpha)**2`, FFT bin 4.
- `d3hrotating`: `B + A*cos(3*phi - 3*alpha)**2`, FFT bin 6.
- Legacy FFT phase recovery divides phase by six and resolves the sign through
  reconstruction agreement. These are fixed bin indices, not a general harmonic
  mapper for arbitrary angle grids. Use acquisition sampling consistent with
  the original full-turn convention. Fitting evaluates the supplied angles.
- Curve fitting retains the original signal normalization, bounded multi-start
  fitting, and optional differential-evolution fallback. Normalization by the
  sampled minimum can produce small amplitude/background bias even on ideal
  signals when their exact minimum is not sampled. This behavior is preserved.
- Orientations use the original 60-degree symmetry convention. R² gates use
  strict `>` comparisons. A0 masks use the whole analysis image and window centers.
- Window maps have shape `(height-window_size+1, width-window_size+1)`, with
  stride one. POI sampling and local map statistics retain the original edge
  handling and coordinate offsets.
- Twist pairs are ordered. Selected pixels use
  `delta = abs(2*(POI2_mean - orientation)) % 60`, and propagated spread is
  `2*sqrt(POI1_std**2 + POI2_std**2)`. Connected intensity masks define the
  eligible region; quality and intensity gates remain active.

The arithmetic class's inherited `sigma_threshold` field does not define pixel
selection in the implemented intensity-mask workflow. Historical descriptions
of selection by an orientation sigma interval are inaccurate for this code path.

`--temp` and `--temp-fix` are retained opt-in legacy adjustments. The latter
alters ROI-A orientation displays and twist maps/histograms; it is not a new
physical calibration. Both default to disabled.

## Exports and caches

CSV fields and the applicable analysis-cache payload structure follow the
original special workflow. The legacy `a_4` field is retained for compatibility;
`harmonic_index` and `a_n` identify the actual harmonic for each model. No claim
is made that historical caches from the full toolkit can be imported.

`--replay-cache` renders a saved cache without input selection or fitting.
`--reuse-curve-cache` loads whole-image fitted window parameters and verifies the
dataset path, cropped input fingerprint, actual angular coordinates, model,
window size, frame count, crop/shape, and A0 percentile/cutoff. Rejected A0 windows
have no reusable fitted values, so changed mask settings require recomputation.
Identity metadata is required. The cached crop is preserved unless
`--ignore-cache-crop` is explicit. That option bypasses crop/shape, cropped-pixel
fingerprint, and derived A0-cutoff checks; the dataset, angles, model, window,
frame count, and percentile must still agree. It is an explicit coordinate-frame
override and should be used only for a cache known to align with the full image.
Use a separate result directory per dataset and model.

## Validation limits

Automated tests exercise numerical recovery, input loading, the complete special
workflow with controlled selections, real figure/table exports, and cache replay.
An independent comparison against the source toolkit checks both models' clean
and noisy signal outputs and window maps. Mouse interaction and appearance on
the target desktop still need manual verification. Real microscopy data has not
yet been provided, so its analysis equivalence is unverified.
