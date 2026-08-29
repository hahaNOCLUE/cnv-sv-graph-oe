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
from cnv_latent_ssm.graph_visibility import (  # noqa: E402
    build_graph_visibility_mask,
    fit_graph_visibility,
    split_visibility_mask,
)
from cnv_latent_ssm.trans_external import fit_trans_external_from_cooler  # noqa: E402

DEFAULT_INPUT = ROOT / "result_2/compartment/chr18_cnv_jcn_balance"
RES = 50_000


def aggregate_500kb(observed, expected, valid, graph_distance,
                    chromosome=None, centromere_start_bp=58_100_000,
                    centromere_end_bp=63_800_000):
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
    if chromosome in ("X", "chrX"):
        centers = (np.arange(n) + 0.5) * factor * RES
        arms = np.full(n, "centromere", dtype=object)
        arms[centers < centromere_start_bp] = "p"
        arms[centers >= centromere_end_bp] = "q"
        pearson = np.full((n, n), np.nan)
        for arm in ("p", "q"):
            indices = np.flatnonzero(arms == arm)
            arm_valid = coarse_valid[indices]
            arm_oe = coarse[np.ix_(indices, indices)]
            arm_pearson = compute_interaction_profile_correlation(
                arm_oe, arm_valid, log_transform=True)
            pearson[np.ix_(indices, indices)] = arm_pearson
    else:
        arms = np.full(n, "whole", dtype=object)
        pearson = compute_interaction_profile_correlation(
            coarse, coarse_valid, log_transform=True)
    return coarse, pearson, graph_min, arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chrom", default="chr18")
    parser.add_argument(
        "--cool-path", type=Path,
        default=ROOT / "result_2/F1/hic_results/mcool/F1.mcool",
        help="Input MCOOL containing the observed contact map",
    )
    parser.add_argument("--physical-ploidy", type=float, default=2.0,
                        help="Molecule-copy normalization P")
    parser.add_argument("--cnv-reference-ploidy", type=float, default=2.0,
                        help="CNVkit log2-to-CN reference scale")
    parser.add_argument(
        "--genome-cnv", type=Path,
        default=ROOT / "result_2/compartment/CNVkit_F1_50kb/F1_50kb.cbs.cns",
        help="Genome-wide segmented CNV file used for trans collision fitting",
    )
    parser.add_argument("--segment-file", default="F1_chr18.balanced_sequence_cn.tsv")
    parser.add_argument("--walk-nodes-file", default="F1_chr18.gGnome_peel_walk_nodes.tsv")
    parser.add_argument("--min-walk-cn", type=float, default=0.0,
                        help="Discard complete peeled walks below this CN")
    parser.add_argument("--external-decay-npz", type=Path,
                        help="Fixed externally estimated same_decay and collision_floor")
    parser.add_argument("--visibility-min-band-fraction", type=float, default=.9)
    parser.add_argument("--visibility-holdout-fraction", type=float, default=.2)
    parser.add_argument("--visibility-breakpoint-padding", type=int, default=500_000)
    parser.add_argument("--visibility-unresolved-fraction", type=float, default=.05)
    parser.add_argument("--visibility-damping", type=float, default=.5)
    parser.add_argument("--visibility-q-min", type=float, default=.5)
    parser.add_argument("--visibility-q-max", type=float, default=2.)
    parser.add_argument("--visibility-tolerance", type=float, default=.01)
    parser.add_argument("--visibility-max-iterations", type=int, default=100)
    parser.add_argument("--visibility-loop-quantile", type=float, default=.995)
    parser.add_argument("--centromere-start", type=int, default=58_100_000)
    parser.add_argument("--centromere-end", type=int, default=63_800_000)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "copy_flow_additive_oe").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(f"{args.cool_path}::/resolutions/{RES}")
    bins = clr.bins().fetch(args.chrom).reset_index(drop=True)
    raw = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    if "weight" in bins.columns:
        weights = pd.to_numeric(bins.weight, errors="coerce").to_numpy()
    else:
        weights = np.ones(len(bins), dtype=float)
    segments = pd.read_csv(input_dir / args.segment_file, sep="\t")
    segments = segments[segments.chrom.eq(args.chrom)].copy()
    centers = (bins.start.to_numpy() + bins.end.to_numpy()) / 2
    cn = np.full(len(bins), np.nan)
    for row in segments.itertuples():
        cn[(centers >= row.start) & (centers < row.end)] = row.balanced_sequence_cn
    valid = np.isfinite(weights) & (weights > 0) & np.isfinite(cn) & (cn >= .1)
    nodes = pd.read_csv(input_dir / args.walk_nodes_file, sep="\t")
    nodes = nodes[nodes.walk_cn >= args.min_walk_cn].copy()
    if nodes.empty:
        raise ValueError("no peeled walks remain after --min-walk-cn filtering")
    ploidy = args.physical_ploidy
    baseline = valid & (np.abs(cn - ploidy) <= max(0.25 * ploidy, 0.25))
    baseline_decay = estimate_native_decay(
        raw, valid, background_mask=np.outer(baseline, baseline))
    cool_path = str(args.cool_path)
    trans_collision_floor, trans_diagnostics = fit_trans_external_from_cooler(
        cool_path, RES,
        str(args.genome_cnv),
        ploidy=args.cnv_reference_ploidy)
    trans_diagnostics.to_csv(output_dir / "chr18.trans_external_fit_by_chrom_pair.tsv",
                             sep="\t", index=False)
    if args.external_decay_npz:
        external_decay = np.load(args.external_decay_npz)
        same_decay = np.asarray(external_decay["same_decay"], float)
        if len(same_decay) < len(raw):
            same_decay = np.pad(same_decay, (0, len(raw) - len(same_decay)),
                                mode="edge")
        same_decay = same_decay[:len(raw)]
        intra_collision_floor = float(external_decay["collision_floor"])
    else:
        same_decay, intra_collision_floor = fit_joint_same_decay(
            raw, valid, np.outer(baseline, baseline), trans_collision_floor,
            ploidy=ploidy)
    kappa = intra_collision_floor / trans_collision_floor
    oe, result = compute_copy_flow_additive_oe(
        raw, bins, valid, cn, nodes, same_decay, RES,
        collision_floor=intra_collision_floor, ploidy=ploidy)

    excluded_visibility = np.zeros(len(bins), bool)
    breakpoint_pair_exclusion = np.zeros_like(raw, bool)
    junction_path = input_dir / "F1_chr18.balanced_junction_cn.tsv"
    if junction_path.exists():
        junctions = pd.read_csv(junction_path, sep="\t")
        intrachrom = junctions[
            junctions.chrom1.astype(str).eq(args.chrom)
            & junctions.chrom2.astype(str).eq(args.chrom)]
        for row in intrachrom.itertuples():
            left = np.abs(centers - float(row.pos1)) <= args.visibility_breakpoint_padding
            right = np.abs(centers - float(row.pos2)) <= args.visibility_breakpoint_padding
            breakpoint_pair_exclusion |= (np.outer(left, right)
                                          | np.outer(right, left))
    source_path = input_dir / "F1_chr18.source_slack_cn.tsv"
    if source_path.exists():
        source = pd.read_csv(source_path, sep="\t")
        source_fraction = (source.groupby("segment_id").source_cn.max()
                           / segments.set_index("segment_id").balanced_sequence_cn)
        unresolved_ids = source_fraction[
            source_fraction > args.visibility_unresolved_fraction].index
        for row in segments[segments.segment_id.isin(unresolved_ids)].itertuples():
            excluded_visibility |= (centers >= row.start) & (centers < row.end)
    # hg38 chrX centromere/assembly gap; bins without an external validity
    # weight must still be excluded from technical-visibility estimation.
    if args.chrom in ("X", "chrX"):
        excluded_visibility |= (centers >= 58_100_000) & (centers < 63_800_000)

    visibility_candidate_mask = build_graph_visibility_mask(
        raw, result.expected, result.visibility_band_cis_expected, valid,
        excluded_bins=excluded_visibility,
        excluded_pairs=breakpoint_pair_exclusion,
        minimum_band_fraction=args.visibility_min_band_fraction,
        enrichment_quantile=args.visibility_loop_quantile,
    )
    visibility_mask, visibility_holdout_mask = split_visibility_mask(
        visibility_candidate_mask, args.visibility_holdout_fraction)
    visibility_fit = fit_graph_visibility(
        raw, result.expected, visibility_mask, valid,
        damping=args.visibility_damping,
        q_min=args.visibility_q_min, q_max=args.visibility_q_max,
        tolerance=args.visibility_tolerance,
        max_iterations=args.visibility_max_iterations,
    )
    visibility_mask = visibility_fit.fit_mask
    q = visibility_fit.visibility
    final_pair_factor = visibility_fit.scale * np.outer(q, q)
    final_expected = final_pair_factor * result.expected
    final_cis_expected = final_pair_factor * result.cis_expected
    final_external_expected = final_pair_factor * result.external_expected
    usable = np.outer(valid, valid) & (final_expected > 0)
    oe = np.divide(raw, final_expected, out=np.zeros_like(raw), where=usable)
    coarse, pearson, graph_distance_500kb, pearson_arms = aggregate_500kb(
        raw, final_expected, valid, result.min_graph_distance_bins,
        chromosome=args.chrom,
        centromere_start_bp=args.centromere_start,
        centromere_end_bp=args.centromere_end)
    output_stem = args.chrom.replace("chr", "chr", 1)
    np.savez_compressed(output_dir / f"{output_stem}.copy_flow_additive.npz", oe=oe,
                        expected=final_expected,
                        graph_expected=result.expected,
                        cis_expected=final_cis_expected,
                        external_expected=final_external_expected,
                        graph_cis_expected=result.cis_expected,
                        graph_external_expected=result.external_expected,
                        visibility_band_cis_expected=result.visibility_band_cis_expected,
                        cis_copy_pairs=result.cis_copy_pairs,
                        external_copy_pairs=result.external_copy_pairs,
                        total_copy_pairs=result.total_copy_pairs,
                        collision_floor=result.collision_floor,
                        oe_500kb=coarse, pearson_500kb=pearson,
                        pearson_arms=pearson_arms,
                        centromere_start_bp=args.centromere_start,
                        centromere_end_bp=args.centromere_end,
                        baseline_decay=baseline_decay,
                        same_molecule_decay=same_decay,
                        min_graph_distance_bins=result.min_graph_distance_bins,
                        min_graph_distance_500kb_bins=graph_distance_500kb,
                        trans_collision_floor=trans_collision_floor,
                        intra_collision_kappa=kappa,
                        min_walk_cn=args.min_walk_cn,
                        capture_visibility=q,
                        global_scale=visibility_fit.scale,
                        visibility_fit_mask=visibility_mask,
                        visibility_holdout_mask=visibility_holdout_mask,
                        visibility_candidate_mask=visibility_candidate_mask,
                        visibility_excluded_bins=excluded_visibility,
                        visibility_supported_bins=visibility_fit.supported_bins,
                        visibility_converged=visibility_fit.converged,
                        visibility_iterations=visibility_fit.iterations,
                        visibility_max_abs_log_ratio=visibility_fit.max_abs_log_ratio,
                        cn=cn, valid=valid, ploidy=ploidy,
                        native_baseline_bins=baseline.sum())
    nodes.to_csv(output_dir / f"{output_stem}.peeled_walk_nodes.tsv", sep="\t", index=False)
    extent=(0, bins.end.max()/1e6, bins.end.max()/1e6, 0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    logoe = np.log2(np.where(oe > 0, oe, np.nan))
    axes[0, 0].imshow(logoe, cmap="RdBu_r", vmin=-2, vmax=2, extent=extent,
                      interpolation="none")
    axes[0, 0].set_title("copy-flow additive log2(O/E)")
    axes[0, 1].imshow(pearson, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                      interpolation="none")
    pearson_title = ("500-kb Pearson (p/q arms separately)"
                     if args.chrom in ("X", "chrX") else "500-kb Pearson")
    axes[0, 1].set_title(pearson_title)
    cis_fraction=np.divide(result.cis_copy_pairs, result.total_copy_pairs,
                           out=np.zeros_like(result.cis_copy_pairs),
                           where=result.total_copy_pairs > 0)
    axes[1, 0].imshow(cis_fraction, cmap="Blues", vmin=0, vmax=1, extent=extent,
                      interpolation="none")
    axes[1, 0].set_title("same-molecule copy-pair fraction M/D")
    delta=np.log2(np.divide(final_cis_expected, final_external_expected,
                            out=np.full_like(final_expected, np.nan),
                            where=final_external_expected > 0))
    axes[1, 1].imshow(delta, cmap="PuOr_r", vmin=-4, vmax=4, extent=extent,
                      interpolation="none")
    axes[1, 1].set_title("log2(cis expected / external expected)")
    for ax in axes.flat:
        ax.set_xlabel(f"{args.chrom} position (Mb)"); ax.set_ylabel(f"{args.chrom} position (Mb)")
    fig.savefig(output_dir / f"{output_stem}.copy_flow_additive.png", dpi=220)

    fig_main, main_axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    main_axes[0].imshow(logoe, cmap="RdBu_r", vmin=-2, vmax=2, extent=extent,
                        interpolation="none")
    main_axes[0].set_title("graph-aware log2(O/E)")
    main_axes[1].imshow(pearson, cmap="RdBu_r", vmin=-1, vmax=1, extent=extent,
                        interpolation="none")
    main_axes[1].set_title(pearson_title)
    for ax in main_axes:
        ax.set_xlabel(f"{args.chrom} position (Mb)")
        ax.set_ylabel(f"{args.chrom} position (Mb)")
    fig_main.savefig(output_dir / f"{output_stem}.oe_and_pearson.png", dpi=220)

    history = pd.DataFrame(
        visibility_fit.history,
        columns=["iteration", "global_scale", "max_abs_log_row_ratio"],
    )
    history.to_csv(output_dir / f"{output_stem}.graph_visibility_history.tsv",
                   sep="\t", index=False)
    visibility_table = bins[["chrom", "start", "end"]].copy()
    visibility_table["q"] = q
    visibility_table["cn"] = cn
    visibility_table["valid"] = valid
    visibility_table["fit_supported"] = visibility_fit.supported_bins
    visibility_table["excluded"] = excluded_visibility
    visibility_table.to_csv(
        output_dir / f"{output_stem}.graph_visibility.tsv", sep="\t", index=False)
    def row_ratio_cv(mask, expected):
        row_observed = np.where(mask, raw, 0).sum(axis=1)
        row_expected = np.where(mask, expected, 0).sum(axis=1)
        row_ratio = np.divide(row_observed, row_expected,
                              out=np.full(len(q), np.nan), where=row_expected > 0)
        use = np.isfinite(row_ratio) & (row_expected > 0)
        return (float(np.nanstd(row_ratio[use]) /
                      max(np.nanmean(row_ratio[use]), 1e-12)), row_ratio, use)

    row_cv, row_ratio, row_use = row_ratio_cv(visibility_mask, final_expected)
    baseline_scale = float(raw[visibility_mask].sum()
                           / max(result.expected[visibility_mask].sum(), 1e-12))
    holdout_cv_before, _, holdout_use_before = row_ratio_cv(
        visibility_holdout_mask, baseline_scale * result.expected)
    holdout_cv_after, _, holdout_use_after = row_ratio_cv(
        visibility_holdout_mask, final_expected)
    supported = visibility_fit.supported_bins
    q_cn_corr = float(np.corrcoef(np.log(q[supported]), cn[supported])[0, 1])
    eigvals, eigvecs = np.linalg.eigh(np.nan_to_num(pearson, nan=0.0))
    pc1_500 = eigvecs[:, np.argmax(eigvals)]
    q_500 = np.array([
        np.nanmean(q[i:i + 10][valid[i:i + 10]])
        if valid[i:i + 10].any() else np.nan
        for i in range(0, len(q), 10)
    ])
    pc_use = np.isfinite(q_500) & np.isfinite(pc1_500)
    pc1_q_corr = (float(np.corrcoef(q_500[pc_use], pc1_500[pc_use])[0, 1])
                  if pc_use.sum() > 2 else np.nan)
    fig_q, q_axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)
    q_axes[0].plot(centers / 1e6, q, lw=.8)
    q_axes[0].axhline(1, color="grey", ls="--", lw=.8)
    q_axes[0].fill_between(centers / 1e6, args.visibility_q_min,
                           args.visibility_q_max, where=excluded_visibility,
                           color="lightgrey", alpha=.5, step="mid")
    q_axes[0].set(ylabel="technical visibility q", title="Graph-aware visibility")
    q_axes[1].plot(history.iteration, history.max_abs_log_row_ratio, marker="o", ms=2)
    q_axes[1].axhline(args.visibility_tolerance, color="grey", ls="--", lw=.8)
    q_axes[1].set(xlabel="iteration", ylabel="max |log row ratio|", yscale="log")
    fig_q.savefig(output_dir / f"{output_stem}.graph_visibility_qc.png", dpi=220)
    pd.Series({
        "converged": visibility_fit.converged,
        "iterations": visibility_fit.iterations,
        "global_scale": visibility_fit.scale,
        "fit_pixel_count": int(visibility_mask.sum()),
        "holdout_pixel_count": int(visibility_holdout_mask.sum()),
        "fit_supported_bins": int(supported.sum()),
        "excluded_bins": int(excluded_visibility.sum()),
        "q_min_observed": float(q[supported].min()),
        "q_max_observed": float(q[supported].max()),
        "row_ratio_cv": row_cv,
        "holdout_supported_bins_before": int(holdout_use_before.sum()),
        "holdout_supported_bins_after": int(holdout_use_after.sum()),
        "holdout_row_ratio_cv_q1": holdout_cv_before,
        "holdout_row_ratio_cv_fitted_q": holdout_cv_after,
        "q_at_lower_bound_fraction": float(
            np.mean(q[supported] <= args.visibility_q_min * (1 + 1e-6))),
        "q_at_upper_bound_fraction": float(
            np.mean(q[supported] >= args.visibility_q_max * (1 - 1e-6))),
        "corr_log_q_cn": q_cn_corr,
        "corr_pc1_q_signed": pc1_q_corr,
        "corr_pc1_q_abs": abs(pc1_q_corr),
    }).to_csv(output_dir / f"{output_stem}.graph_visibility_qc.tsv",
              sep="\t", header=False)
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
    print(output_dir / f"{output_stem}.copy_flow_additive.png")
    print(output_dir / f"{output_stem}.oe_and_pearson.png")
    print(output_dir / "chr18.external_fit_contact_by_distance.png")
    print("ploidy", ploidy, "native_baseline_bins", int(baseline.sum()))
    print("trans_collision_floor", trans_collision_floor)
    print("intra_collision_floor", result.collision_floor, "kappa", kappa)
    print("graph_visibility_scale", visibility_fit.scale,
          "iterations", visibility_fit.iterations,
          "converged", visibility_fit.converged)
    print("graph_visibility_row_cv", row_cv,
          "holdout_cv_q1", holdout_cv_before,
          "holdout_cv_fitted_q", holdout_cv_after,
          "corr_log_q_cn", q_cn_corr,
          "abs_corr_pc1_q", abs(pc1_q_corr))


if __name__ == "__main__":
    main()
