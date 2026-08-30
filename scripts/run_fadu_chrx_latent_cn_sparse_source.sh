#!/usr/bin/env bash
set -euo pipefail

project=/home/dell/a1/microc
repo="$project/code/cnv_latent_ssm"
source_root="$project/Fadu/graph_oe_continuous_neoloop_cnv"
run_root="$source_root/latent_cn_sparse_source"
cn_dir="$run_root/continuous_cnv"
balance_dir="$run_root/balance"
oe_dir="$run_root/oe_physical_only_minwalk0.1_armwise"
mkdir -p "$cn_dir" "$balance_dir" "$oe_dir"

python_bin=/home/dell/miniconda3/envs/EagleC2/bin/python
r_cnvkit=/home/dell/miniconda3/envs/cnvkit/bin/Rscript
r_jabba=/home/dell/miniconda3/envs/jabba/bin/Rscript
profile="$project/Fadu/eaglec2_chrX_10kb_prob0.1/cnv_correction/FaDu_50000.CNV-profile.bedGraph"
cnr="$cn_dir/FaDu_50kb.continuous.cnr"
cns="$cn_dir/FaDu_50kb.continuous.cns"

"$python_bin" "$repo/scripts/neoloop_profile_to_continuous_cnr.py" \
  --profile "$profile" --output "$cnr" \
  --exclude-region X:58100000-63800000 \
  --outliers "$cn_dir/FaDu_50kb.excluded_outlier_bins.tsv"

"$r_cnvkit" "$repo/scripts/segment_cnr_cnvkit_like.R" \
  "$cnr" "$cns" "$cn_dir/FaDu_50kb.cbs.raw.tsv" 1e-4 2

"$python_bin" "$repo/scripts/balance_chr18_cnv_jcn.py" \
  --cnv "$cns" --cnr "$cnr" \
  --sv "$source_root/joint_evidence/FaDu_chrX.selected.SV_calls.txt" \
  --interchrom-sv "$source_root/empty_interchrom.SV_calls.txt" \
  --junction-counts "$source_root/joint_evidence/FaDu_chrX.selected.local_junction_counts.tsv" \
  --one-copy-junction-pairs 700.765774417 \
  --cool-uri "$project/Fadu/GSM6463428_FaDu_WT_1.mcool::/resolutions/50000" \
  --reference-window-bins 5 \
  --centromere-start 58100000 --centromere-end 63800000 \
  --fix-parent-cn \
  --chrom X --ploidy 2 --cnv-snap-tolerance 100000 \
  --cnv-snap-min-jump 0.25 --outdir "$balance_dir"

# Always emit a standalone CNV/JCN overview in addition to the balance QC.
MPLCONFIGDIR=/tmp/mplconfig "$python_bin" "$repo/scripts/plot_chr_cnv_sv_overview.py" \
  --chrom X --cnv "$cns" --cnr "$cnr" \
  --junctions "$balance_dir/F1_chr18.balanced_junction_cn.tsv" \
  --ploidy 2 --centromere-start 58100000 --centromere-end 63800000 \
  --output "$balance_dir/X.cnv_jcn_overview.png"

"$r_jabba" "$repo/scripts/run_chr18_ggnome_peel.R" "$balance_dir"

"$python_bin" "$repo/scripts/run_local_copy_flow_additive_chr18.py" \
  --input-dir "$balance_dir" --output-dir "$oe_dir" --chrom X \
  --cool-path "$project/Fadu/GSM6463428_FaDu_WT_1.mcool" \
  --physical-ploidy 2 --cnv-reference-ploidy 2 --genome-cnv "$cns" \
  --min-walk-cn 0.1 \
  --external-decay-npz "$project/result_2/compartment2/chrX_50kb_CNV_refinedSV/external_decay/autosome_copy_neutral_decay.npz" \
  --visibility-q-min 1 --visibility-q-max 1 \
  --centromere-start 58100000 --centromere-end 63800000
