#!/usr/bin/env python3
"""Run CALDER on one chromosome arm from an ICE-balanced COOL matrix."""
import argparse
import gzip
from pathlib import Path
import subprocess

import cooler
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cool-uri", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--start-bp", type=int, required=True)
    p.add_argument("--end-bp", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--genome", default="hg38")
    p.add_argument("--cores", type=int, default=4)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(args.cool_uri)
    cool_chrom = args.chrom if args.chrom in clr.chromnames else f"chr{args.chrom}"
    if cool_chrom not in clr.chromnames:
        raise ValueError(f"{args.chrom} is absent from {args.cool_uri}")
    matrix = np.asarray(clr.matrix(balance=True).fetch(cool_chrom), float)
    lo = max(0, args.start_bp // clr.binsize)
    hi = min(len(matrix), (args.end_bp + clr.binsize - 1) // clr.binsize)
    if hi <= lo:
        raise ValueError("arm interval contains no bins")

    dump = args.output_dir / f"{args.chrom}.{args.label}.{clr.binsize}.balanced.tsv.gz"
    with gzip.open(dump, "wt") as handle:
        for i in range(lo, hi):
            values = matrix[i, i:hi]
            js = np.flatnonzero(np.isfinite(values) & (values > 0)) + i
            for j in js:
                handle.write(f"{i * clr.binsize}\t{j * clr.binsize}\t{matrix[i, j]:.8g}\n")

    chrom_key = args.chrom.removeprefix("chr")
    r_file = args.output_dir / "run_calder.R"
    r_file.write_text(f'''suppressPackageStartupMessages(library(CALDER))
CALDER(
  contact_file_dump=list("{chrom_key}"="{dump}"),
  chrs="{chrom_key}",
  bin_size={clr.binsize},
  genome="{args.genome}",
  save_dir="{args.output_dir}",
  save_intermediate_data=TRUE,
  n_cores={args.cores},
  sub_domains=FALSE
)
''')
    with (args.output_dir / "calder.log").open("w") as log:
        completed = subprocess.run([args.rscript, str(r_file)], stdout=log,
                                   stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(f"CALDER failed; see {args.output_dir / 'calder.log'}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
