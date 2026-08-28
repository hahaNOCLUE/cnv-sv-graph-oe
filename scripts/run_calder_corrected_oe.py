#!/usr/bin/env python3
"""Run CALDER on a saved corrected expected matrix at its native resolution."""
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
    p.add_argument("--model-npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--genome", default="hg38")
    p.add_argument("--cores", type=int, default=4)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(args.cool_uri)
    observed = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    model = np.load(args.model_npz)
    expected = np.asarray(model["expected"], float)
    valid = np.asarray(model["valid"], bool)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected matrices have different shapes")

    oe = np.divide(observed, expected, out=np.zeros_like(observed),
                   where=np.isfinite(expected) & (expected > 0))
    eligible = np.outer(valid, valid) & np.isfinite(oe) & (oe > 0)
    dump = args.output_dir / f"{args.chrom}.{clr.binsize}.corrected_oe.tsv.gz"
    with gzip.open(dump, "wt") as handle:
        for i in range(len(oe)):
            js = np.flatnonzero(eligible[i, i:]) + i
            pos_i = i * clr.binsize
            for j in js:
                handle.write(f"{pos_i}\t{j * clr.binsize}\t{oe[i, j]:.8g}\n")

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
    command = [args.rscript, str(r_file)]
    with (args.output_dir / "calder.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(f"CALDER failed; see {args.output_dir / 'calder.log'}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
