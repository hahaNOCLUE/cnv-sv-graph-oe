#!/usr/bin/env python3
"""Local chr18 evaluation of the generic oriented copy-flow expected model."""
from pathlib import Path
import argparse
import sys

import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/dell/a1/microc")
sys.path.insert(0, str(ROOT / "code/cnv_latent_ssm/src"))
from cnv_latent_ssm.graph_expected import (  # noqa: E402
    compute_copy_flow_additive_oe,
    aggregate_observed_expected,
    estimate_native_decay,
    fit_joint_same_decay,
)
from cnv_latent_ssm.features import compute_interaction_profile_correlation  # noqa: E402
from cnv_latent_ssm.trans_external import fit_trans_external_from_cooler  # noqa: E402

DEFAULT_INPUT = ROOT / "result_2/compartment/chr18_cnv_jcn_balance"
RES = 50_000


def aggregate_500kb(observed, expected, valid, graph_distance,
                    min_compartment_distance_bp=2_000_000):
    factor = 10
    n = int(np.ceil(len(valid) / factor))
    size = n * factor
    valid_grid = np.outer(valid, valid)
    _, _, coarse, coarse_valid = aggregate_observed_expected(
        observed, expected, valid, factor)
    graph_pad = np.full((size, size), np.inf)
    graph_pad[:len(valid), :len(valid)] = np.where(
        valid_grid, graph_distance, np.inf)
    graph_min = graph_pad.reshape(n, factor, n, factor).min(axis=(1, 3))
    return coarse, compute_interaction_profile_correlation(
        coarse, coarse_valid, log_transform=True), graph_min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chrom", default="chr18")
    parser.add_argument("--physical-ploidy", type=float, default=2.0,
                        help="Molecule-copy normalization P")
    parser.add_argument("--cnv-reference-ploidy", type=float, default=2.0,
                        help="CNVkit log2-to-CN reference scale")
    parser.add_argument("--min-walk-cn", type=float, default=0.0,
                        help="Discard complete peeled walks below this CN")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "copy_flow_additive_oe").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(f"{ROOT}/result_2/F1/hic_results/mcool/F1.mcool::/resolutions/{RES}")
    bins = clr.bins().fetch(args.chrom).reset_index(drop=True)
    raw = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    weights = pd.to_numeric(bins.weight, errors="coerce").to_numpy()
    segments = pd.read_csv(input_dir / "F1_chr18.balanced_sequence_cn.tsv", sep="\t")
    centers = (bins.start.to_numpy() + bins.end.to_numpy()) / 2
    cn = np.full(len(bins), np.nan)
    for row in segments.itertuples():
        cn[(centers >= row.start) & (centers < row.end)] = row.balanced_sequence_cn
    valid = np.isfinite(weights) & (weights > 0) & np.isfinite(cn) & (cn >= .1)
    nodes = pd.read_csv(input_dir / "F1_chr18.gGnome_peel_walk_nodes.tsv", sep="\t")
    nodes = nodes[nodes.walk_cn >= args.min_walk_cn].copy()
    if nodes.empty:
        raise ValueError("no peeled walks remain after --min-walk-cn filtering")
    ploidy = args.physical_ploidy
    baseline = valid & (np.abs(cn - ploidy) <= max(0.25 * ploidy, 0.25))
    baseline_decay = estimate_native_decay(
        raw, valid, background_mask=np.outer(baseline, baseline))
    cool_path = str(ROOT / "result_2/F1/hic_results/mcool/F1.mcool")
    trans_collision_floor, trans_diagnostics = fit_trans_external_from_cooler(
        cool_path, RES,
        str(ROOT / "result_2/compartment/CNVkit_F1_50kb/F1_50kb.cbs.cns"),
        ploidy=args.cnv_reference_ploidy)
    trans_diagnostics.to_csv(output_dir / "chr18.trans_external_fit_by_chrom_pair.tsv",
                             sep="\t", index=False)
    same_decay, intra_collision_floor = fit_joint_same_decay(
        raw, valid, np.outer(baseline, baseline), trans_collision_floor,
        ploidy=ploidy)
    kappa = intra_collision_floor / trans_collision_floor
    oe, result = compute_copy_flow_additive_oe(
        raw, bins, valid, cn, nodes, same_decay, RES,
        collision_floor=intra_collision_floor, ploidy=ploidy)
    coarse, pearson, graph_distance_500kb = aggregate_500kb(
        raw, result.expected, valid, result.min_graph_distance_bins)
    np.savez_compressed(output_dir / "chr18.copy_flow_additive.npz", oe=oe,
                        expected=result.expected,
                        cis_expected=result.cis_expected,
                        external_expected=result.external_expected,
                        cis_copy_pairs=result.cis_copy_pairs,
                        external_copy_pairs=result.external_copy_pairs,
                        total_copy_pairs=result.total_copy_pairs,
                        collision_floor=result.collision_floor,
                        oe_500kb=coarse, pearson_500kb=pearson,
                        baseline_decay=baseline_decay,
                        same_molecule_decay=same_decay,
                        min_graph_distance_bins=result.min_graph_distance_bins,
                        min_graph_distance_500kb_bins=graph_distance_500kb,
                        trans_collision_floor=trans_collision_floor,
                        intra_collision_kappa=kappa,
                        min_walk_cn=args.min_walk_cn,
                        cn=cn, valid=valid, ploidy=ploidy,
                        native_baseline_bins=baseline.sum())
    nodes.to_csv(output_dir / "chr18.peeled_walk_nodes.tsv", sep="\t", index=False)
    extent=(0, bins.end.max()/1e6, bins.end.max()/1e6, 0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    logoe = np.log2(np.where(oe > 0, oe, np.nan))
    axes[0, 0].imshow(logoe, cmap="RdBu_r", vmin=-2, vmax=2, extent=extent,
                      interpolation="none")
    axes[0, 0].set_title("copy-flow additive log2(O/E)")
    axes[0, 1].imshow(pearson, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                      interpolation="none")
    axes[0, 1].set_title("500-kb Pearson")
    cis_fraction=np.divide(result.cis_copy_pairs, result.total_copy_pairs,
                           out=np.zeros_like(result.cis_copy_pairs),
                           where=result.total_copy_pairs > 0)
    axes[1, 0].imshow(cis_fraction, cmap="Blues", vmin=0, vmax=1, extent=extent,
                      interpolation="none")
    axes[1, 0].set_title("same-molecule copy-pair fraction M/D")
    delta=np.log2(np.divide(result.cis_expected, result.external_expected,
                            out=np.full_like(result.expected, np.nan),
                            where=result.external_expected > 0))
    axes[1, 1].imshow(delta, cmap="PuOr_r", vmin=-4, vmax=4, extent=extent,
                      interpolation="none")
    axes[1, 1].set_title("log2(cis expected / external expected)")
    for ax in axes.flat:
        ax.set_xlabel(f"{args.chrom} position (Mb)"); ax.set_ylabel(f"{args.chrom} position (Mb)")
    fig.savefig(output_dir / "chr18.copy_flow_additive.png", dpi=220)
    distances, medians, means, eligible_counts, nonzero_counts = [], [], [], [], []
    for distance in range(1, len(raw)):
        observed = np.diag(raw, k=distance)
        external_only = np.diag(result.cis_copy_pairs <= 0, k=distance)
        usable = (external_only & valid[:-distance] & valid[distance:]
                  & np.isfinite(observed) & (observed >= 0))
        if usable.any():
            selected = observed[usable]
            distances.append(distance)
            medians.append(float(np.median(selected)))
            means.append(float(np.mean(selected)))
            eligible_counts.append(int(usable.sum()))
            nonzero_counts.append(int(np.count_nonzero(selected)))
    diagnostic = pd.DataFrame({
        "distance_bins": distances,
        "distance_bp": np.asarray(distances) * RES,
        "median_observed": medians,
        "mean_observed": means,
        "eligible_pair_count": eligible_counts,
        "nonzero_pair_count": nonzero_counts,
        "nonzero_fraction": np.divide(nonzero_counts, eligible_counts),
    })
    diagnostic.to_csv(output_dir / "chr18.external_fit_contact_by_distance.tsv",
                      sep="\t", index=False)
    fig2, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(diagnostic.distance_bp / 1e6, diagnostic.mean_observed,
            lw=1.5, label="mean observed (including zeros)")
    ax.set_yscale("log")
    ax.set_xlabel("reference distance (Mb)")
    ax.set_ylabel("external-mask mean observed contact")
    ax.set_title("External-fit mask diagnostic")
    ax2 = ax.twinx()
    ax2.plot(diagnostic.distance_bp / 1e6, diagnostic.nonzero_fraction,
             color="grey", alpha=.35, lw=1, label="nonzero fraction")
    ax2.set_ylabel("nonzero fraction")
    fig2.savefig(output_dir / "chr18.external_fit_contact_by_distance.png", dpi=220)
    print(output_dir / "chr18.copy_flow_additive.png")
    print(output_dir / "chr18.external_fit_contact_by_distance.png")
    print("ploidy", ploidy, "native_baseline_bins", int(baseline.sum()))
    print("trans_collision_floor", trans_collision_floor)
    print("intra_collision_floor", result.collision_floor, "kappa", kappa)


if __name__ == "__main__":
    main()
