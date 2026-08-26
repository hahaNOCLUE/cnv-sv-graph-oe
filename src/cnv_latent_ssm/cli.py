"""CLI Module: Command line entry point for CNV-aware Latent Chromatin State Model."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .caller import run_cnv_latent_ssm
from .io import (
    export_cnv_bedgraph,
    export_compartment_bedgraph,
    export_compartment_state_bed,
    export_local_artifact_bedgraph,
    export_parameters_tsv,
    export_rank_compartment_bedgraph,
)
from .plot import plot_ssm_summary
from .qc import compute_qc_metrics, export_qc_tsv


def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CNV-aware Latent Chromatin State Model for robust continuous compartment calling."
    )
    # Inputs
    parser.add_argument("--cool", required=True, help="Path to .cool or .mcool file")
    parser.add_argument("--chrom", required=True, help="Chromosome name (e.g. chr8)")
    parser.add_argument("--res", type=int, default=50000, help="Resolution in bp (default: 50000)")
    
    # CNV options
    parser.add_argument(
        "--cnv-mode",
        choices=["external", "none"],
        default="external",
        help="CNV mode: external (from external CNV file, default) or none (c_i=0)",
    )
    parser.add_argument("--cnv-file", default=None, help="Path to external CNV bed/tsv file")
    parser.add_argument(
        "--cnv-value-type", choices=["copy_number", "log2_ratio"], default="copy_number",
        help="Meaning of the external value column; CNVkit .cns uses log2_ratio",
    )
    parser.add_argument(
        "--cnv-effect-scale", choices=["log2", "linear"], default="log2",
        help="CNV covariate used by the SSM: log2(CN/P) or linear CN/P-1 (default: log2)",
    )
    parser.add_argument("--ploidy", type=float, default=2.0, help="Sample baseline ploidy P (default: 2.0)")
    parser.add_argument(
        "--min-cn",
        type=float,
        default=0.2,
        help="Copy number threshold below which regions are treated as homozygous/extreme deletions (default: 0.2)",
    )
    parser.add_argument(
        "--no-mask-deletions",
        action="store_true",
        default=False,
        help="Disable automatic masking of homozygous deletion regions (default: False, deletions are masked)",
    )
    
    # Phasing & reference options
    parser.add_argument(
        "--phasing-file", default=None,
        help="GC/gene track for sign orientation and optional --loading-init phasing",
    )
    parser.add_argument("--phasing-col", default=None, help="Column name in phasing file to use")
    
    # Model parameters
    parser.add_argument("--n-pcs", type=int, default=10, help="Number of observation PCs K (default: 10)")
    parser.add_argument(
        "--model-type",
        choices=["ar1", "local_linear_trend", "two_scale"],
        default="ar1",
        help="SSM formulation: ar1, local_linear_trend, or two_scale (global + local artifact)",
    )
    parser.add_argument("--max-iter", type=int, default=400, help="Max EM iterations (default: 400)")
    parser.add_argument("--tol", type=float, default=1e-5, help="EM convergence tolerance (default: 1e-5)")
    parser.add_argument(
        "--loading-init", choices=["pc1", "uniform", "phasing"], default="pc1",
        help="Initial compartment loading; phasing explicitly uses the supplied GC/gene track as an initialization prior",
    )
    parser.add_argument(
        "--freeze-cnv-effect", action="store_true", default=False,
        help="Estimate CN effects before EM and keep G fixed during latent-state fitting",
    )
    parser.add_argument(
        "--sv-file", default=None,
        help="EagleC/EagleC2 SV calls; decouple linear AR transitions at paired SV endpoints",
    )
    parser.add_argument(
        "--sv-edge-strength", type=float, default=0.0,
        help="Strength of graph transitions between paired SV endpoints (default: 0, disabled)",
    )
    parser.add_argument(
        "--sv-decouple-endpoints", action="store_true", default=False,
        help="Also cut ordinary linear AR transitions at SV endpoints (default: keep them)",
    )
    parser.add_argument(
        "--sv-cnv-weighted", action="store_true", default=False,
        help="Weight paired SV transitions by CNVkit-derived endpoint dosage",
    )
    parser.add_argument("--contact-graph-k", type=int, default=0,
                        help="Distal contact-profile neighbors per bin for global graph state")
    parser.add_argument("--contact-graph-min-distance-bins", type=int, default=40,
                        help="Minimum genomic separation for contact graph edges")
    parser.add_argument("--contact-graph-min-correlation", type=float, default=0.3,
                        help="Minimum O/E profile Pearson r for a graph edge (default: 0.3)")
    parser.add_argument("--contact-graph-non-mutual", action="store_true", default=False,
                        help="Use the union of directed top-k neighbors instead of mutual-kNN")
    parser.add_argument("--contact-graph-signed", action="store_true", default=False,
                        help="Use positive same-state and negative opposite-state profile edges")
    parser.add_argument("--contact-graph-strength", type=float, default=0.0,
                        help="Precision multiplier for the global contact graph")
    parser.add_argument("--sv-distance-oe", action="store_true", default=False,
                        help="Use confidence/CN-weighted single-SV graph distances in O/E expected")
    parser.add_argument("--sv-max-hops", type=int, default=0,
                        help="Maximum SV edges per distance path; 0 means unlimited (default: 0)")
    parser.add_argument("--is-microc", action="store_true", default=True, help="Micro-C flag (default: True)")
    parser.add_argument("--balance", action="store_true", default=False, help="Use balanced contacts (default: False)")
    parser.add_argument(
        "--no-decouple-breakpoints",
        action="store_true",
        default=False,
        help="Disable breakpoint decoupling across gaps/deletions (default: False, breakpoints are decoupled)",
    )
    parser.add_argument(
        "--breakpoint-min-gap-bins", type=int, default=3,
        help="Restart after at least this many consecutive invalid bins (default: 3)",
    )
    parser.add_argument(
        "--decouple-cnv-breakpoints", action="store_true", default=False,
        help="Restart the latent state at large CNV jumps (default: off)",
    )
    parser.add_argument(
        "--cnv-breakpoint-threshold", type=float, default=0.4,
        help="Absolute delta-log2(CN) used with --decouple-cnv-breakpoints (default: 0.4)",
    )
    parser.add_argument("--outdir", default="./output", help="Output directory")
    parser.add_argument("--prefix", default=None, help="Output file prefix (default: auto-generated)")
    parser.add_argument("--no-plot", action="store_true", help="Skip summary plot generation")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose logging")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    os.makedirs(args.outdir, exist_ok=True)
    if args.prefix is None:
        sample_name = os.path.basename(args.cool).split(".")[0]
        prefix = f"{sample_name}_{args.chrom}_{args.res // 1000}kb_{args.model_type}_{args.cnv_mode}"
    else:
        prefix = args.prefix

    out_base = os.path.join(args.outdir, prefix)

    logger.info("=== Starting CNV-aware Latent Chromatin State Model ===")
    logger.info(f"Target: {args.cool} ({args.chrom} @ {args.res}bp)")
    logger.info(f"CNV Mode: {args.cnv_mode} (ploidy={args.ploidy}, min_cn={args.min_cn}), Model: {args.model_type}")

    results, extra_data = run_cnv_latent_ssm(
        cooler_path=args.cool,
        chrom=args.chrom,
        resolution=args.res,
        cnv_mode=args.cnv_mode,
        cnv_file=args.cnv_file,
        cnv_value_type=args.cnv_value_type,
        cnv_effect_scale=args.cnv_effect_scale,
        ploidy=args.ploidy,
        min_cn_threshold=args.min_cn,
        mask_deletions=not args.no_mask_deletions,
        phasing_file=args.phasing_file,
        phasing_col=args.phasing_col,
        n_pcs=args.n_pcs,
        model_type=args.model_type,
        max_iter=args.max_iter,
        tol=args.tol,
        loading_init=args.loading_init,
        freeze_cnv_effect=args.freeze_cnv_effect,
        sv_file=args.sv_file,
        sv_edge_strength=args.sv_edge_strength,
        sv_decouple_endpoints=args.sv_decouple_endpoints,
        sv_cnv_weighted=args.sv_cnv_weighted,
        contact_graph_k=args.contact_graph_k,
        contact_graph_min_distance_bins=args.contact_graph_min_distance_bins,
        contact_graph_min_correlation=args.contact_graph_min_correlation,
        contact_graph_mutual=not args.contact_graph_non_mutual,
        contact_graph_signed=args.contact_graph_signed,
        contact_graph_strength=args.contact_graph_strength,
        sv_distance_oe=args.sv_distance_oe,
        sv_max_hops=None if args.sv_max_hops == 0 else args.sv_max_hops,
        is_microc=args.is_microc,
        balance=args.balance,
        decouple_breakpoints=not args.no_decouple_breakpoints,
        breakpoint_min_gap_bins=args.breakpoint_min_gap_bins,
        decouple_cnv_breakpoints=args.decouple_cnv_breakpoints,
        cnv_breakpoint_threshold=args.cnv_breakpoint_threshold,
        verbose=args.verbose,
    )

    # Export outputs
    score_bedgraph = f"{out_base}.compartment_score.bedGraph"
    state_bed = f"{out_base}.compartment_state.bed"
    cnv_bedgraph = f"{out_base}.cnv.bedGraph"
    param_tsv = f"{out_base}.parameters.tsv"
    qc_tsv = f"{out_base}.qc.tsv"

    logger.info(f"Exporting compartment score to {score_bedgraph}...")
    export_compartment_bedgraph(results, score_bedgraph)
    rank_score_bedgraph = f"{out_base}.compartment_score_rank_minus1_1.bedGraph"
    logger.info(f"Exporting CALDER-style rank score to {rank_score_bedgraph}...")
    export_rank_compartment_bedgraph(results, rank_score_bedgraph)

    if results.local_artifact_score is not None:
        local_bedgraph = f"{out_base}.local_artifact_score.bedGraph"
        logger.info(f"Exporting local artifact score to {local_bedgraph}...")
        export_local_artifact_bedgraph(results, local_bedgraph)

    logger.info(f"Exporting discrete states and uncertainty to {state_bed}...")
    export_compartment_state_bed(results, state_bed)

    logger.info(f"Exporting CNV deviation to {cnv_bedgraph}...")
    export_cnv_bedgraph(results, cnv_bedgraph)

    logger.info(f"Exporting estimated parameters to {param_tsv}...")
    export_parameters_tsv(results, param_tsv)

    sv_edge_table = extra_data.get("sv_edge_table")
    if sv_edge_table is not None and not sv_edge_table.empty:
        sv_edge_tsv = f"{out_base}.sv_edge_cn_weights.tsv"
        logger.info(f"Exporting CNV-weighted SV edges to {sv_edge_tsv}...")
        sv_edge_table.to_csv(sv_edge_tsv, sep="\t", index=False)

    # Compute & Export QC
    qc_dict = compute_qc_metrics(results, extra_data)
    logger.info(f"Exporting QC metrics to {qc_tsv}...")
    export_qc_tsv(qc_dict, qc_tsv)

    # Plotting
    if not args.no_plot:
        logger.info("Generating multi-track summary plot...")
        plot_path = plot_ssm_summary(
            results=results,
            extra_data=extra_data,
            out_prefix=out_base,
            title=f"CNV-aware Latent Chromatin State Model ({args.model_type.upper()}) — {args.chrom}",
        )
        logger.info(f"Saved plot: {plot_path}")

    logger.info("=== CNV-aware Latent Chromatin State Modeling Complete ===")
    print("\n--- Summary QC Metrics ---")
    for k, v in qc_dict.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
