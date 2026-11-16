# Sample data

Place each experiment in its own subfolder, for example:

```text
sample_data/
  experiment/
    images/
      000.tif
      001.tif
      ...
    metadata.dat
```

A single multi-frame TIFF is also supported. Keep the frame order, acquisition
angle origin, angle increment, scan span, and analyzer configuration with the data.
Select `--d3hfixed` or `--d3hrotating` to match the actual acquisition.

```powershell
uv run d3h-polarization --images sample_data/experiment/images --dat-file sample_data/experiment/metadata.dat --twist-angle
```

This folder includes 721 TIFF images and 3 acquisition DAT files supplied with
the project (approximately 1.22 GB). The original filenames and folder structure
are preserved. Operating-system metadata files are excluded.

Confirm the acquisition settings before selecting a model; folder names alone
do not establish the angle axis or analyzer configuration. Inclusion of these
samples does not establish analysis equivalence with the original toolkit.
Tests generate their own small synthetic inputs in temporary directories;
generated results are never committed.
