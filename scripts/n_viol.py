#!/usr/bin/env python3
"""
Upper-limit consistency check for the archived NMMA fits of SN 2021ugl.

For each model, counts the upper limits (ULs) for which the MAP light curve
is brighter than the limiting magnitude by more than sigma_eb:

    N_viol = #{ ULs : m_lim - m_model(t_UL) > sigma_eb }

It also recomputes the per-band residual statistic X^2_red (Equation 6 of
the paper) as a check that the MAP light curves are read correctly: the
values must match the 'chi2_dict' stored by NMMA and Tables 5 and 6.

Usage (from the repository root):
    python scripts/n_viol.py
"""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "SN2021ugl_NMMA_posteriors"
CANDNAME = "SN_2021ugl"


def iso_to_mjd(iso: str) -> float:
    return (
        datetime.fromisoformat(iso) - datetime(1858, 11, 17)
    ).total_seconds() / 86400.0


def load_photometry(config: str):
    """Rows of the fitted NMMA data file: (mjd, band, mag, err); err = inf for ULs."""
    rows = []
    for line in (DATA / config / "data" / f"{CANDNAME}.dat").read_text().splitlines():
        if not line.strip():
            continue
        iso, band, mag, err = line.split()
        rows.append((iso_to_mjd(iso), band, float(mag), float(err)))
    return rows


def n_viol(config: str, model: str):
    """Return (N_viol, N_UL, X2_red per band) for one archived fit."""
    mdir = DATA / config / model
    manifest = json.loads((mdir / f"{model}_{CANDNAME}_manifest.json").read_text())
    t0 = manifest["options"]["trigger_time"]
    sigma_eb = manifest["options"]["error_budget"]

    best = json.loads(
        Path(glob.glob(str(mdir / "*bestfit_params.json"))[0]).read_text()
    )
    t_grid = np.asarray(best["obs_times"])  # days since trigger_time
    mags = best["Magnitudes"]  # MAP light curve per band

    n_v, n_ul, x2 = 0, 0, {}
    for t, band, mag, err in load_photometry(config):
        if band not in mags:
            continue
        m_model = np.interp(t - t0, t_grid, mags[band])
        if np.isinf(err):  # upper limit
            n_ul += 1
            n_v += (mag - m_model) > sigma_eb
        else:  # detection
            x2.setdefault(band, []).append(
                (mag - m_model) ** 2 / (err**2 + sigma_eb**2)
            )
    return int(n_v), n_ul, {b: float(np.mean(v)) for b, v in x2.items()}


if __name__ == "__main__":
    for config in ("early_time_with_upper_limits", "full_baseline_with_upper_limits"):
        print(f"\n{config}")
        for mdir in sorted(
            p for p in (DATA / config).iterdir() if p.is_dir() and p.name != "data"
        ):
            nv, nul, x2 = n_viol(config, mdir.name)
            x2s = "  ".join(f"{b}={v:.2f}" for b, v in x2.items())
            print(f"  {mdir.name:16s} N_viol/N_UL = {nv}/{nul}   X2_red: {x2s}")
