# Local sample data

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

These images and DAT files stay local and are ignored by Git. No real sample
dataset has been supplied yet. Tests generate their own small synthetic inputs
in temporary directories; generated results are never committed.
