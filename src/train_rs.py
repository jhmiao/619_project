"""
Compatibility entrypoint for randomized-smoothing DFL training.

The shared implementation lives in train_dfl.py. Running this file trains only
the RS variant and preserves the historical output defaults.
"""

from __future__ import annotations

import sys

from train_dfl import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--models", "rs"]
    main()
