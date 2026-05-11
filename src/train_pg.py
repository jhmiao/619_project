"""
Compatibility entrypoint for perturbation-gradient DFL training.

The shared implementation lives in train_dfl.py. Running this file trains only
the PG variant and preserves the historical output defaults, including all PG
estimators unless --single-estimator-only is passed.
"""

from __future__ import annotations

import sys

from train_dfl import main


def _translate_legacy_args(argv: list[str]) -> list[str]:
    translated = ["--models", "pg"]
    skip_next = False

    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--train-all-estimators":
            continue
        if arg == "--mse-weight":
            translated.append("--pg-mse-weight")
        elif arg.startswith("--mse-weight="):
            translated.append(arg.replace("--mse-weight", "--pg-mse-weight", 1))
        elif arg == "--fairness-weight":
            translated.append("--pg-fairness-weight")
        elif arg.startswith("--fairness-weight="):
            translated.append(arg.replace("--fairness-weight", "--pg-fairness-weight", 1))
        else:
            translated.append(arg)

        if arg in {"--mse-weight", "--fairness-weight"} and index + 1 < len(argv):
            translated.append(argv[index + 1])
            skip_next = True

    return translated


if __name__ == "__main__":
    sys.argv[1:] = _translate_legacy_args(sys.argv[1:])
    main()
