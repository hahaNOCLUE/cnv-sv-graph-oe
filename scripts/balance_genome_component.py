#!/usr/bin/env python3
"""Balance the genome-wide SV component containing a target chromosome."""
from __future__ import annotations
import argparse
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

ORIENTATIONS = ("++", "+-", "-+", "--")


def arguments():
    root = Path("/home/dell/a1/microc")
    p = argparse.ArgumentParser()
    p.add_argument("--target-chrom", default="chrX")
    p.add_argument("--cnv", type=Path, default=root / "result_2/compartment/CNVkit_F1_50kb/F1_50kb.cbs.cns")
    p.add_argument("--sv", type=Path, default=root / "result_2/EagleC2/F1_genomewide_cis_trans_50kb/F1_genomewide_50kb_EagleC2_SVs_raw.SV_calls.txt")
    p.add_argument("--cool-uri", default=str(root / "result_2/F1/hic_results/mcool/F1.mcool") + "::/resolutions/50000")
    p.add_argument("--outdir", type=Path, default=root / "result_2/compartment/chrX_genome_component_graph")
    p.add_argument("--cn-reference-ploidy", type=float, default=2.0)
    p.add_argument("--flow-weight", type=float, default=1e8)
    p.add_argument("--source-l1", type=float, default=10.0)
    p.add_argument("--jcn-prior-weight", type=float, default=2.0)
    p.add_argument("--support-weight", type=float, default=1.0)
    p.add_argument("--junction-beta", type=float, default=117.75908651,
                   help="One-copy breakpoint support calibrated from sampled normal loci")
    return p.parse_args()


def chromosome_component(sv, target):
    adjacency = {}
    for row in sv.itertuples():
        adjacency.setdefault(row.chrom1, set()).add(row.chrom2)
        adjacency.setdefault(row.chrom2, set()).add(row.chrom1)
    seen, stack = set(), [target]
    while stack:
        chrom = stack.pop()
        if chrom in seen:
            continue
        seen.add(chrom)
        stack.extend(adjacency.get(chrom, ()) - seen)
    return seen


def local_support(clr, chrom1, pos1, strand1, chrom2, pos2, strand2):
    bins = clr.bins()[:]
    offsets = {c: clr.offset(c) for c in clr.chromnames}
    sizes = dict(zip(clr.chromnames, clr.chromsizes))
    resolution = clr.binsize
    local_i = int(pos1 // resolution) - (strand1 == "+")
    local_j = int(pos2 // resolution) - (strand2 == "+")
    i = offsets[chrom1] + min(max(local_i, 0),
                              int(np.ceil(sizes[chrom1] / resolution)) - 1)
    j = offsets[chrom2] + min(max(local_j, 0),
                              int(np.ceil(sizes[chrom2] / resolution)) - 1)
    matrix = clr.matrix(balance=False)
    observed = float(np.asarray(matrix[i, j]).item())
    background = []
    for shift in range(-10, 11):
        if shift == 0:
            continue
        ii, jj = i + shift, j + shift
        if (0 <= ii < len(bins) and 0 <= jj < len(bins)
                and bins.iloc[ii].chrom == chrom1
                and bins.iloc[jj].chrom == chrom2):
            value = matrix[ii, jj]
            if np.isfinite(value) and value >= 0:
                background.append(float(np.asarray(value).item()))
    return observed, float(np.mean(background)) if background else 0.0


def main():
    a = arguments()
    a.outdir.mkdir(parents=True, exist_ok=True)
    sv = pd.read_csv(a.sv, sep="\t")
    component = chromosome_component(sv, a.target_chrom)
    sv = sv[sv.chrom1.isin(component) & sv.chrom2.isin(component)].copy().reset_index(drop=True)
    probabilities = sv.loc[:, ORIENTATIONS].astype(float)
    sv["orientation"] = probabilities.idxmax(axis=1)
    sv["eaglec2_probability"] = probabilities.max(axis=1)
    sv["sv_id"] = [f"SV{i + 1:04d}" for i in range(len(sv))]

    cns = pd.read_csv(a.cnv, sep="\t", comment="#")
    cns = cns[cns.chromosome.isin(component)].copy()
    cns["sequence_cn"] = a.cn_reference_ploidy * np.exp2(cns.log2.astype(float))
    cuts = {chrom: set() for chrom in component}
    for row in cns.itertuples():
        cuts[row.chromosome].update((int(row.start), int(row.end)))
    for row in sv.itertuples():
        cuts[row.chrom1].add(int(row.pos1)); cuts[row.chrom2].add(int(row.pos2))
    records = []
    for row in cns.sort_values(["chromosome", "start"]).itertuples():
        local = sorted(x for x in cuts[row.chromosome]
                       if int(row.start) <= x <= int(row.end))
        for start, end in zip(local, local[1:]):
            if start < end:
                records.append({"segment_id": f"SEQ{len(records)+1:05d}",
                                "chrom": row.chromosome, "start": start,
                                "end": end, "sequence_cn": row.sequence_cn})
    segments = pd.DataFrame(records)
    segment_lookup = {(r.chrom, int(r.start), int(r.end)): i
                      for i, r in enumerate(segments.itertuples())}
    nseg, nv = len(segments), 2 * len(segments)

    def endpoint(chrom, pos, strand):
        local = segments[segments.chrom.eq(chrom)]
        if strand == "+":
            hit = local[local.end.le(int(pos))]
            i = int(hit.index[-1]) if len(hit) else int(local.index[0])
            return 2 * i + 1, i, "R"
        hit = local[local.start.ge(int(pos))]
        i = int(hit.index[0]) if len(hit) else int(local.index[-1])
        return 2 * i, i, "L"

    ref_rows = []
    for chrom, group in segments.groupby("chrom", sort=False):
        ids = group.index.to_numpy()
        for left, right in zip(ids[:-1], ids[1:]):
            if segments.loc[left, "end"] == segments.loc[right, "start"]:
                ref_rows.append((left, right, int(segments.loc[right, "start"])))
    nr, nj = len(ref_rows), len(sv)
    off_j, off_source = nr, nr + nj
    incident = [[] for _ in range(nv)]
    for edge, (left, right, _) in enumerate(ref_rows):
        incident[2 * left + 1].append(edge)
        incident[2 * right].append(edge)
    edge_vertices, cn_targets = [], []
    seq_cn = segments.sequence_cn.to_numpy(float)
    for j, row in sv.iterrows():
        v1, b1, s1 = endpoint(row.chrom1, row.pos1, row.orientation[0])
        v2, b2, s2 = endpoint(row.chrom2, row.pos2, row.orientation[1])
        incident[v1].append(off_j + j); incident[v2].append(off_j + j)
        edge_vertices.append((v1, v2, b1, b2, s1, s2))
        capacity = min(seq_cn[b1], seq_cn[b2])
        def cn_jump(chrom, pos):
            local = segments[segments.chrom.eq(chrom)]
            left = local[local.end.le(int(pos))]
            right = local[local.start.ge(int(pos))]
            if not len(left) or not len(right):
                return 0.0
            return abs(seq_cn[int(left.index[-1])] - seq_cn[int(right.index[0])])
        cn_targets.append(min(max(cn_jump(row.chrom1, row.pos1),
                                  cn_jump(row.chrom2, row.pos2)), capacity))

    # Only true chromosome ends and CN discontinuities may retain unresolved
    # endpoint mass. Every known interchromosomal breakpoint is a real edge.
    source_kind = np.full(nv, "unresolved", dtype=object)
    for chrom, group in segments.groupby("chrom", sort=False):
        source_kind[2 * int(group.index[0])] = "telomere"
        source_kind[2 * int(group.index[-1]) + 1] = "telomere"
    for left, right, _ in ref_rows:
        if abs(seq_cn[left] - seq_cn[right]) >= .25:
            source_kind[2 * left + 1] = "unresolved"
            source_kind[2 * right] = "unresolved"
    source_allowed = np.ones(nv, dtype=bool)

    clr = cooler.Cooler(a.cool_uri)
    support = [local_support(clr, r.chrom1, r.pos1, r.orientation[0],
                             r.chrom2, r.pos2, r.orientation[1])
               for r in sv.itertuples()]
    support_y = np.array([x[0] for x in support])
    support_eta = np.array([x[1] for x in support])
    # A one-copy derivative adjacency should have the same order of contact as
    # an ordinary one-bin reference adjacency. Estimate this scale genome-wide.
    adjacent_rates = []
    matrix = clr.matrix(balance=False)
    for chrom in component:
        lo, hi = clr.extent(chrom)
        values = np.asarray(matrix[lo:hi-1, lo+1:hi].diagonal(), float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values):
            adjacent_rates.extend(values.tolist())
    adjacent_beta = float(np.median(adjacent_rates))
    beta = a.junction_beta

    rows, cols, vals = [], [], []
    for vertex in range(nv):
        for edge in incident[vertex]:
            rows.append(vertex); cols.append(edge); vals.append(1.0)
    incidence = sparse.csr_matrix((vals, (rows, cols)),
                                  shape=(nv, off_source)).toarray()
    side_cn = np.repeat(seq_cn, 2)
    upper = np.full(off_source, np.inf)
    for edge, (left, right, _) in enumerate(ref_rows):
        upper[edge] = min(seq_cn[left], seq_cn[right])
    for j, (_, _, b1, b2, _, _) in enumerate(edge_vertices):
        upper[off_j + j] = min(seq_cn[b1], seq_cn[b2])
    probability = sv.eaglec2_probability.to_numpy(float)
    cn_targets = np.asarray(cn_targets)
    support_target = np.minimum(
        np.maximum((support_y - support_eta) / beta, 0),
        upper[off_j:off_source])

    # Sparse LP with exact mass conservation. Two non-negative deviation
    # variables encode each absolute residual around the support-derived JCN
    # and the weaker CN-step prior. Poisson variance determines support weight.
    edge0, source0 = 0, off_source
    support_plus0, support_minus0 = source0 + nv, source0 + nv + nj
    prior_plus0, prior_minus0 = source0 + nv + 2 * nj, source0 + nv + 3 * nj
    nvar = source0 + nv + 4 * nj
    objective = np.zeros(nvar)
    objective[source0:source0 + nv] = np.where(
        source_kind == "unresolved", a.source_l1, 0.0)
    support_penalty = a.support_weight * beta / np.sqrt(support_y + 1.0)
    objective[support_plus0:support_minus0] = support_penalty
    objective[support_minus0:prior_plus0] = support_penalty
    prior_penalty = a.jcn_prior_weight * probability
    objective[prior_plus0:prior_minus0] = prior_penalty
    objective[prior_minus0:nvar] = prior_penalty

    eq_rows, eq_cols, eq_values, eq_rhs = [], [], [], []
    for vertex in range(nv):
        row = len(eq_rhs)
        for edge in np.flatnonzero(incidence[vertex]):
            eq_rows.append(row); eq_cols.append(edge0 + edge); eq_values.append(1.0)
        eq_rows.append(row); eq_cols.append(source0 + vertex); eq_values.append(1.0)
        eq_rhs.append(side_cn[vertex])
    for j in range(nj):
        row = len(eq_rhs)
        eq_rows.extend((row, row, row))
        eq_cols.extend((off_j + j, support_plus0 + j, support_minus0 + j))
        eq_values.extend((1.0, -1.0, 1.0)); eq_rhs.append(support_target[j])
        row = len(eq_rhs)
        eq_rows.extend((row, row, row))
        eq_cols.extend((off_j + j, prior_plus0 + j, prior_minus0 + j))
        eq_values.extend((1.0, -1.0, 1.0)); eq_rhs.append(cn_targets[j])
    equality = sparse.csr_matrix(
        (eq_values, (eq_rows, eq_cols)), shape=(len(eq_rhs), nvar))
    bounds = [(0, float(u)) for u in upper] + [(0, None)] * (nvar - off_source)
    fit = linprog(objective, A_eq=equality, b_eq=np.asarray(eq_rhs),
                  bounds=bounds, method="highs")
    if not fit.success:
        raise RuntimeError(fit.message)
    ref_cn = fit.x[:nr]
    jcn = fit.x[off_j:off_source]
    source = fit.x[source0:source0 + nv]
    residual = incidence @ fit.x[:off_source] + source - side_cn

    segments["balanced_sequence_cn"] = segments.sequence_cn
    segments.to_csv(a.outdir / "F1_component.balanced_sequence_cn.tsv", sep="\t", index=False)
    pd.DataFrame({"edge_id": [f"REF{i+1:05d}" for i in range(nr)],
                  "chrom": [segments.loc[l, "chrom"] for l, _, _ in ref_rows],
                  "boundary": [b for _, _, b in ref_rows],
                  "reference_cn": ref_cn}).to_csv(
        a.outdir / "F1_component.balanced_reference_cn.tsv", sep="\t", index=False)
    junction = sv[["sv_id", "chrom1", "pos1", "chrom2", "pos2", "orientation",
                   "eaglec2_probability"]].copy()
    junction["vertex1"] = [f"{r.chrom1}:{int(r.pos1)}:{edge_vertices[i][4]}"
                           for i, r in enumerate(sv.itertuples())]
    junction["vertex2"] = [f"{r.chrom2}:{int(r.pos2)}:{edge_vertices[i][5]}"
                           for i, r in enumerate(sv.itertuples())]
    junction["junction_cn"] = jcn
    junction["endpoint_cn_capacity"] = cn_targets
    junction["support_observed"] = support_y
    junction["support_background"] = support_eta
    junction["support_beta_per_cn"] = beta
    junction["support_expected"] = support_eta + beta * jcn
    junction.to_csv(a.outdir / "F1_component.balanced_junction_cn.tsv", sep="\t", index=False)
    source_table = pd.DataFrame({
        "segment_id": np.repeat(segments.segment_id, 2),
        "chrom": np.repeat(segments.chrom, 2),
        "side": np.tile(["L", "R"], nseg), "source_cn": source,
        "source_class": source_kind})
    source_table.to_csv(a.outdir / "F1_component.source_cn.tsv", sep="\t", index=False)
    with (a.outdir / "F1_component.balance_summary.txt").open("w") as handle:
        handle.write(f"target_chrom\t{a.target_chrom}\ncomponent_chromosomes\t{len(component)}\n")
        handle.write(f"segments\t{nseg}\nreference_edges\t{nr}\nsv_edges\t{nj}\n")
        handle.write(f"interchrom_sv_edges\t{int((sv.chrom1 != sv.chrom2).sum())}\n")
        handle.write(f"junction_support_beta\t{beta:.10g}\n")
        handle.write(f"reference_adjacent_beta_diagnostic\t{adjacent_beta:.10g}\n")
        handle.write(f"max_flow_residual_cn\t{np.max(np.abs(residual)):.10g}\n")
        handle.write(f"total_junction_cn\t{jcn.sum():.10g}\n")
        for kind in ("telomere", "unresolved", "forbidden"):
            handle.write(f"source_{kind}_cn\t{source[source_kind == kind].sum():.10g}\n")
    print(a.outdir)


if __name__ == "__main__":
    main()
