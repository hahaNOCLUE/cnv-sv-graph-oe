#!/usr/bin/env python3
"""Sensitivity scan for graph-aware CAIC-like CN responses on chrX."""
import argparse
from pathlib import Path
import sys

import cooler
import numpy as np
import pandas as pd

ROOT = Path("/home/dell/a1/microc")
sys.path.insert(0, str(ROOT / "code/cnv_latent_ssm/src"))
from cnv_latent_ssm.cn_response import fit_monotonic_cn_response  # noqa: E402


def ratio(observed, expected, mask):
    return float(observed[mask].sum() / expected[mask].sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cool-uri", default=str(ROOT / "result_2/F1/hic_results/mcool/F1.mcool") + "::/resolutions/50000")
    p.add_argument("--chrom", default="chrX")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.npz)
    raw = np.asarray(cooler.Cooler(args.cool_uri).matrix(balance=False).fetch(args.chrom), float)
    valid = z["valid"].astype(bool)
    cn = z["cn"] / float(z["ploidy"])
    cis, ext = z["cis_expected"], z["external_expected"]
    total = cis + ext
    n = len(cn)
    row_cn = np.broadcast_to(cn[:, None], (n, n)).ravel()
    col_cn = np.broadcast_to(cn[None, :], (n, n)).ravel()
    upper = np.triu(np.outer(valid, valid), 1) & np.isfinite(raw) & (total > 0)
    distance = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    # Avoid self/near-diagonal local structure and require a high-confidence
    # component assignment for each response fit.
    cis_train = upper & (distance >= 40) & (cis >= 4 * ext)
    ext_train = upper & (distance >= 200) & (ext >= 4 * cis)
    rows = []
    for ridge in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        # The fit functions receive all flattened pixels; refit masks are
        # enforced by zeroing exposure outside the selected training strata.
        # Refit explicitly with masked exposure to avoid locus identity terms.
        cis_fit = fit_monotonic_cn_response(
            raw.ravel(), np.where(cis_train, cis, 0).ravel(), row_cn, col_cn,
            ridge=ridge, smooth=ridge)
        ext_fit = fit_monotonic_cn_response(
            raw.ravel(), np.where(ext_train, ext, 0).ravel(), row_cn, col_cn,
            ridge=ridge, smooth=ridge, seed=180427)
        gc, ge = cis_fit.evaluate(cn), ext_fit.evaluate(cn)
        corrected = cis * np.outer(gc, gc) + ext * np.outer(ge, ge)
        cis_dom, ext_dom = upper & (cis > ext), upper & (ext >= cis)
        rows.append({
            "ridge": ridge, "R_all": ratio(raw, corrected, upper),
            "R_cis": ratio(raw, corrected, cis_dom),
            "R_ext": ratio(raw, corrected, ext_dom),
            "cis_direction": cis_fit.direction,
            "ext_direction": ext_fit.direction,
            "g_cis_min": gc[valid].min(), "g_cis_max": gc[valid].max(),
            "g_ext_min": ge[valid].min(), "g_ext_max": ge[valid].max(),
            "cis_objective": cis_fit.objective,
            "ext_objective": ext_fit.objective,
        })
        np.savez_compressed(
            args.output_dir / f"chrX.cn_response.ridge_{ridge:g}.npz",
            expected=corrected, g_cis=gc, g_ext=ge,
            cis_knots=cis_fit.knots, cis_log_values=cis_fit.values,
            ext_knots=ext_fit.knots, ext_log_values=ext_fit.values,
            ridge=ridge, valid=valid, cn=cn)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "chrX.cn_response_sensitivity.tsv",
                 sep="\t", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
