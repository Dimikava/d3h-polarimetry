"""Run from a checkout or use the installed d3h-polarization command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from d3h_polarimetry.cli.run_polarization import main

if __name__ == "__main__":
    main()
