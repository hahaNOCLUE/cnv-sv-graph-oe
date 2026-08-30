#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outliers", type=Path)
    parser.add_argument(
        "--exclude-region", action="append", default=[],
        help="Half-open CHROM:START-END interval to leave missing (repeatable)",
    )
    args = parser.parse_args()

    bins = pd.read_csv(
        args.profile, sep="\t", header=None,
        names=["chromosome", "start", "end", "relative_cn"],
    )
    bins["relative_cn"] = pd.to_numeric(bins.relative_cn, errors="coerce")
    excluded = []
    for value in args.exclude_region:
        chromosome, interval = value.rsplit(":", 1)
        start, end = map(int, interval.split("-", 1))
        excluded.append((chromosome, start, end))
    invalid = ~np.isfinite(bins.relative_cn) | bins.relative_cn.le(0)
    bins["is_excluded_region"] = False
    for chromosome, start, end in excluded:
        region = (bins.chromosome.astype(str).eq(chromosome)
                  & bins.end.gt(start) & bins.start.lt(end))
        bins.loc[region, "is_excluded_region"] = True
        invalid |= region
    bins.loc[invalid, "relative_cn"] = np.nan
    bins["log2"] = np.log2(bins.relative_cn)
    bins["is_outlier"] = False
    bins["imputed_log2"] = bins["log2"]
    for _, index in bins.groupby("chromosome", sort=False).groups.items():
        values = bins.loc[index, "log2"]
        local_median = values.rolling(21, center=True, min_periods=5).median()
        upper_deviation = values - local_median
        absolute_deviation = (values - local_median).abs()
        local_mad = absolute_deviation.rolling(21, center=True, min_periods=5).median()
        # Only remove isolated upward technical spikes. Sustained amplification
        # and low-copy/deleted bins are biological CN input and must remain.
        threshold = np.maximum(6 * 1.4826 * local_mad.fillna(0), 3.5)
        flagged = upper_deviation > threshold
        bins.loc[index, "is_outlier"] = flagged.to_numpy()
        bins.loc[index, "imputed_log2"] = values.where(~flagged, local_median).to_numpy()
    if args.outliers:
        bins.loc[bins.is_outlier].to_csv(args.outliers, sep="\t", index=False)
    bins["log2"] = bins.imputed_log2
    complete = []
    for chromosome, local in bins.groupby("chromosome", sort=False):
        local = local.sort_values("start").set_index("start")
        bin_size = int(np.median(local.end.to_numpy() - local.index.to_numpy()))
        starts = np.arange(int(local.index.min()), int(local.index.max()) + 1, bin_size)
        local = local.reindex(starts)
        missing = local.chromosome.isna()
        local["chromosome"] = chromosome
        local["end"] = np.where(missing, starts + bin_size, local.end)
        local["is_missing_bin"] = missing
        # Fill only isolated missing bins. Long assembly gaps (especially
        # centromeres) must remain absent rather than becoming a linear CN
        # ramp between unrelated flanking observations.
        local["log2"] = local.log2.interpolate(
            limit=2, limit_direction="both", limit_area="inside")
        local["is_excluded_region"] = local.is_excluded_region.fillna(False)
        local.loc[local.is_excluded_region, "log2"] = np.nan
        local["relative_cn"] = local.relative_cn.where(
            np.isfinite(local.relative_cn), np.exp2(local.log2))
        local.loc[local.is_excluded_region, "relative_cn"] = np.nan
        local["is_outlier"] = local.is_outlier.fillna(False)
        local["imputed_log2"] = local.log2
        local.index.name = "start"
        complete.append(local.reset_index())
    bins = pd.concat(complete, ignore_index=True)
    bins["gene"] = "-"
    bins["depth"] = bins.relative_cn
    bins["weight"] = np.where(np.isfinite(bins.log2), 1.0, 0.0)
    bins[["chromosome", "start", "end", "gene", "log2", "depth", "weight"]].to_csv(
        args.output, sep="\t", index=False
    )


if __name__ == "__main__":
    main()
