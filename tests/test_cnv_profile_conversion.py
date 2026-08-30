import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_excluded_gap_is_not_interpolated(tmp_path):
    profile = tmp_path / "profile.bedgraph"
    output = tmp_path / "profile.cnr"
    pd.DataFrame([
        ("X", 0, 50_000, 1.0),
        ("X", 50_000, 100_000, 2.0),
        ("X", 100_000, 150_000, 0.0),
        ("X", 150_000, 200_000, 0.0),
        ("X", 200_000, 250_000, 8.0),
        ("X", 250_000, 300_000, 1.0),
    ]).to_csv(profile, sep="\t", header=False, index=False)
    script = (Path(__file__).parents[1] / "scripts"
              / "neoloop_profile_to_continuous_cnr.py")
    subprocess.run([
        sys.executable, str(script), "--profile", str(profile),
        "--output", str(output), "--exclude-region", "X:100000-250000",
    ], check=True)
    result = pd.read_csv(output, sep="\t")
    gap = result[(result.start >= 100_000) & (result.end <= 250_000)]
    assert len(gap) == 3
    assert not np.isfinite(gap.log2).any()
    assert (gap.weight == 0).all()

