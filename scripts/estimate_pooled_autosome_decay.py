#!/usr/bin/env python3
"""Estimate a pooled copy-neutral cis decay from autosomes."""
import argparse
from pathlib import Path
import sys

import cooler
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path("/home/dell/a1/microc")
sys.path.insert(0, str(ROOT / "code/cnv_latent_ssm/src"))
from cnv_latent_ssm.graph_expected import _weighted_nonincreasing_isotonic
from cnv_latent_ssm.trans_external import fit_trans_external_from_cooler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cool", required=True)
    p.add_argument("--cnv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=50000)
    p.add_argument("--ploidy", type=float, default=2.0)
    p.add_argument("--exclude", default="chr18")
    p.add_argument("--max-bins", type=int, default=3200)
    args = p.parse_args()
    clr = cooler.Cooler(f"{args.cool}::/resolutions/{args.resolution}")
    cnv = pd.read_csv(args.cnv, sep="\t", comment="#")
    chrom_col = "chromosome" if "chromosome" in cnv else "chrom"
    start_col, end_col = "start", "end"
    value_col = "log2" if "log2" in cnv else "seg.mean"
    totals = np.zeros(args.max_bins)
    counts = np.zeros(args.max_bins)
    used = []
    excluded = set(args.exclude.split(","))
    for chrom in [f"chr{i}" for i in range(1, 23) if f"chr{i}" not in excluded]:
        try:
            bins = clr.bins().fetch(chrom).reset_index(drop=True)
            matrix = np.asarray(clr.matrix(balance=False).fetch(chrom), float)
        except Exception:
            continue
        centers = (bins.start.to_numpy() + bins.end.to_numpy()) / 2
        cn = np.full(len(bins), np.nan)
        table = cnv[cnv[chrom_col].eq(chrom)]
        for row in table.itertuples(index=False):
            d = row._asdict()
            cn[(centers >= d[start_col]) & (centers < d[end_col])] = (
                args.ploidy * 2 ** float(d[value_col]))
        weight = pd.to_numeric(bins.weight, errors="coerce").to_numpy()
        valid = np.isfinite(weight) & (weight > 0) & np.isfinite(cn) & (np.abs(cn-args.ploidy) <= .5)
        for distance in range(1, min(len(bins), args.max_bins)):
            values = np.diag(matrix, distance)
            keep = valid[:-distance] & valid[distance:] & np.isfinite(values) & (values >= 0)
            totals[distance] += values[keep].sum()
            counts[distance] += keep.sum()
        used.append(chrom)
    support = np.flatnonzero((counts > 0) & (np.arange(args.max_bins) > 0))
    rate = totals[support] / counts[support]
    b_trans, _ = fit_trans_external_from_cooler(args.cool, args.resolution,
                                                 str(args.cnv), ploidy=args.ploidy)
    knots = np.linspace(np.log(support.min()), np.log(support.max()), min(24, len(support)))
    logd = np.log(support)
    floor_fraction = 1 - 1 / args.ploidy
    initial = np.maximum(rate - floor_fraction*b_trans, 1e-6)
    logp = np.interp(knots, logd, np.log(initial))
    logp = _weighted_nonincreasing_isotonic(logp, np.ones_like(logp))
    x0 = np.r_[logp, np.log(b_trans)]
    scale = max(counts[support].sum(), 1)
    def objective(x):
        mu = np.exp(np.interp(logd, knots, x[:-1])) + floor_fraction*np.exp(x[-1])
        nll = np.sum(counts[support]*mu - totals[support]*np.log(np.maximum(mu, 1e-12)))
        prior = .5*((x[-1]-np.log(b_trans))/.35)**2
        return (nll+prior)/scale
    constraints = [{"type":"ineq", "fun":lambda x, i=i: x[i]-x[i+1]}
                   for i in range(len(knots)-1)]
    fit = minimize(objective, x0, method="SLSQP", constraints=constraints,
                   bounds=[(-30,30)]*len(knots)+[(-30,10)],
                   options={"maxiter":3000, "ftol":1e-9})
    if not fit.success:
        raise RuntimeError(fit.message)
    all_d = np.arange(args.max_bins)
    same = np.exp(np.interp(np.log(np.maximum(all_d,1)), knots, fit.x[:-1]))
    same[0] = same[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, same_decay=same,
                        collision_floor=np.exp(fit.x[-1]), trans_collision_floor=b_trans,
                        distances=support, totals=totals[support], counts=counts[support],
                        chromosomes=np.asarray(used))
    print("chromosomes", ",".join(used))
    print("B_trans", b_trans, "B_intra", np.exp(fit.x[-1]))

if __name__ == "__main__":
    main()
