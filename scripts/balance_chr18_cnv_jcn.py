#!/usr/bin/env python3
"""Solve shared-yield Poisson JCN with hard CN flow conservation."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

ORIENTATIONS = ("++", "+-", "-+", "--")

def get_args():
    root = Path("/home/dell/a1/microc")
    p = argparse.ArgumentParser()
    p.add_argument("--cnv", type=Path, default=root / "result_2/compartment/CNVkit_F1_50kb/F1_50kb.cbs.cns")
    p.add_argument("--cnr", type=Path, default=root / "result_2/compartment/CNVkit_F1_50kb/F1_50kb.continuous.cnr",
                   help="Pre-CBS CNVkit bins to overlay on the CN panel")
    p.add_argument("--sv", type=Path, default=root / "result_2/EagleC2/F1/F1_chr18_50kb_EagleC2_SVs_raw.SV_calls.txt")
    p.add_argument("--interchrom-sv", type=Path, default=root / "result_2/EagleC2/F1_genomewide_cis_trans_50kb/F1_genomewide_50kb_EagleC2_SVs_raw.SV_calls.txt",
                   help="Genome-wide calls used to add explicit chr18-external edges")
    p.add_argument("--outdir", type=Path, default=root / "result_2/compartment/chr18_cnv_jcn_balance")
    p.add_argument("--chrom", default="chr18")
    p.add_argument("--ploidy", type=float, default=2.0,
                   help="Nominal scale for CNVkit log2 ratio: CN=P*2**log2")
    p.add_argument("--reference-ploidy-cap", type=float, default=None,
                   help="Optional upper bound on every reference-adjacency CN")
    p.add_argument("--breakpoint-reference-ploidy-cap", type=float, default=None,
                   help="Optional reference-CN cap only at SV breakpoint boundaries")
    p.add_argument("--foldback-min-probability", type=float, default=.9)
    p.add_argument("--foldback-dedup-distance", type=int, default=50_000)
    p.add_argument("--junction-counts", type=Path,
                   help="Generic orientation-aware local junction count table")
    p.add_argument("--one-copy-junction-pairs", type=float,
                   help="Expected local excess pairs contributed by JCN=1")
    p.add_argument("--cn-sigma-floor", type=float, default=.15,
                   help="Minimum uncertainty of each latent parent-segment CN")
    p.add_argument("--cn-sigma-scale", type=float, default=1.0,
                   help="Additional CN uncertainty, scaled by 1/sqrt(probes)")
    p.add_argument("--likelihood-tolerance", type=float, default=1.0,
                   help="Allowed likelihood increase during sparse-source stages")
    p.add_argument("--cnv-snap-tolerance",type=int,default=100_000)
    p.add_argument("--cnv-snap-min-jump",type=float,default=.25)
    return p.parse_args()

def endpoint_vertex(pos, strand, starts, ends):
    pos = int(pos)
    if strand == "+":
        found = np.flatnonzero(ends <= pos)
        i = int(found[-1]) if len(found) else 0
        return 2 * i + 1, i, "R"
    found = np.flatnonzero(starts >= pos)
    i = int(found[0]) if len(found) else len(starts) - 1
    return 2 * i, i, "L"

def main():
    a = get_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    cns = pd.read_csv(a.cnv, sep="\t", comment="#")
    parent = cns[cns.chromosome.astype(str).eq(a.chrom)].copy().sort_values("start").reset_index(drop=True)
    if parent.empty: raise ValueError(f"no {a.chrom} records in {a.cnv}")
    parent["chrom"] = parent.chromosome.astype(str)
    parent["copy_number"] = a.ploidy * np.exp2(pd.to_numeric(parent.log2, errors="raise"))
    sv = pd.read_csv(a.sv, sep="\t")
    sv = sv[(sv.chrom1 == a.chrom) & (sv.chrom2 == a.chrom)].copy().reset_index(drop=True)
    probabilities = sv.loc[:, ORIENTATIONS].astype(float)
    sv["orientation"] = probabilities.idxmax(axis=1)
    sv["eaglec2_probability"] = probabilities.max(axis=1)
    sv["sv_id"] = [f"SV{i+1:02d}" for i in range(len(sv))]
    sv["is_external"] = False
    sv["is_foldback"] = (sv.orientation.isin(("++", "--")) &
                         sv.eaglec2_probability.ge(a.foldback_min_probability))
    dropped=[]
    for fb in sv[sv.is_foldback].sort_values("eaglec2_probability", ascending=False).itertuples():
        overlap=(sv.sv_id.ne(fb.sv_id) &
                 (sv.pos1.sub(fb.pos1).abs().le(a.foldback_dedup_distance)) &
                 (sv.pos2.sub(fb.pos2).abs().le(a.foldback_dedup_distance)) &
                 sv.eaglec2_probability.lt(fb.eaglec2_probability))
        for row in sv[overlap].itertuples():
            dropped.append({"dropped_sv_id":row.sv_id,"representative_foldback":fb.sv_id,
                            "dropped_probability":row.eaglec2_probability,
                            "foldback_probability":fb.eaglec2_probability})
        sv=sv[~overlap].copy()
    pd.DataFrame(dropped).to_csv(a.outdir/"F1_chr18.foldback_deduplicated_calls.tsv",
                                 sep="\t",index=False)
    sv=sv.reset_index(drop=True)
    n_intra = len(sv)
    external = pd.read_csv(a.interchrom_sv, sep="\t")
    external = external[(external.chrom1.eq(a.chrom)) ^
                        (external.chrom2.eq(a.chrom))].copy().reset_index(drop=True)
    external_probabilities = external.loc[:, ORIENTATIONS].astype(float)
    external["orientation"] = external_probabilities.idxmax(axis=1)
    external["eaglec2_probability"] = external_probabilities.max(axis=1)
    external["sv_id"] = [f"EXT{i+1:02d}" for i in range(len(external))]
    external["is_external"] = True
    external["is_foldback"] = False
    sv = pd.concat([sv, external], ignore_index=True, sort=False)

    # Snap a nearby CBS boundary to an SV endpoint only when pre-CBS bins show
    # a sustained local CN transition. This treats 50-kb calls at their actual
    # resolution while using one global rule for every endpoint.
    raw_bins=pd.read_csv(a.cnr,sep="\t",comment="#")
    raw_bins=raw_bins[raw_bins.chromosome.astype(str).eq(a.chrom)].copy()
    raw_bins["copy_number"]=a.ploidy*np.exp2(pd.to_numeric(raw_bins.log2,errors="coerce"))
    endpoint_meta={}
    for row in sv.itertuples():
        is_foldback=(not row.is_external) and row.orientation in ("++", "--")
        for chrom,pos in ((row.chrom1,int(row.pos1)),(row.chrom2,int(row.pos2))):
            if chrom != a.chrom:
                continue
            meta=(float(row.eaglec2_probability),int(is_foldback),row.sv_id,row.orientation)
            if pos not in endpoint_meta or meta[:2] > endpoint_meta[pos][:2]:
                endpoint_meta[pos]=meta
    endpoint_values=sorted(endpoint_meta)
    # Build all eligible boundary-endpoint pairs, then greedily retain
    # one-to-one matches. High-confidence calls are preferred first, followed
    # by fold-back-compatible orientations and then endpoint distance.
    candidates=[]
    for i in range(1,len(parent)):
        boundary=int(parent.loc[i,"start"])
        for p in endpoint_values:
            if abs(p-boundary)>a.cnv_snap_tolerance:
                continue
            left=raw_bins[(raw_bins.end<=p)&(raw_bins.end>p-100_000)].copy_number
            right=raw_bins[(raw_bins.start>=p)&(raw_bins.start<p+100_000)].copy_number
            if len(left)<2 or len(right)<2:
                continue
            left_cn=float(np.median(left)); right_cn=float(np.median(right))
            jump=abs(right_cn-left_cn)
            if jump>=a.cnv_snap_min_jump:
                probability,foldback,sv_id,orientation=endpoint_meta[p]
                candidates.append((-probability,-foldback,abs(p-boundary),-jump,
                                   i,boundary,p,left_cn,right_cn,jump,
                                   probability,foldback,sv_id,orientation))
    used_boundaries=set(); used_endpoints=set(); selected=[]
    for _,_,_,_,i,boundary,p,left_cn,right_cn,jump,probability,foldback,sv_id,orientation in sorted(candidates):
        if i in used_boundaries or p in used_endpoints:
            continue
        used_boundaries.add(i); used_endpoints.add(p)
        selected.append((i,boundary,p,left_cn,right_cn,jump,
                         probability,foldback,sv_id,orientation))
    snap_rows=[]
    for i,boundary,p,left_cn,right_cn,jump,probability,foldback,sv_id,orientation in sorted(selected):
        parent.loc[i-1,"end"]=p; parent.loc[i,"start"]=p
        snap_rows.append({"original_cbs_boundary":boundary,"snapped_boundary":p,"distance_bp":p-boundary,
                          "raw_left_median_cn":left_cn,"raw_right_median_cn":right_cn,"raw_cn_jump":jump,
                          "matched_sv_id":sv_id,"matched_orientation":orientation,
                          "matched_probability":probability,"matched_foldback":bool(foldback)})
    if not np.all(parent.end.to_numpy(int) > parent.start.to_numpy(int)):
        raise ValueError("SV snapping produced a non-positive CNV segment")
    pd.DataFrame(snap_rows).to_csv(a.outdir/"F1_chr18.cnv_sv_boundary_matches.tsv",sep="\t",index=False)

    cuts = set(parent.start.astype(int)) | set(parent.end.astype(int))
    cuts |= set(sv.loc[sv.chrom1.eq(a.chrom), "pos1"].astype(int))
    cuts |= set(sv.loc[sv.chrom2.eq(a.chrom), "pos2"].astype(int))
    records = []
    for i, row in parent.iterrows():
        local = sorted(c for c in cuts if int(row.start) <= c <= int(row.end))
        for start, end in zip(local, local[1:]):
            if start < end:
                records.append({"segment_id": f"SEQ{len(records)+1:03d}", "chrom": a.chrom,
                    "start": start, "end": end, "parent_cnv_segment": f"CNSEG{i+1:03d}",
                    "sequence_cn": float(row.copy_number)})
    segments = pd.DataFrame(records)
    starts, ends = segments.start.to_numpy(int), segments.end.to_numpy(int)
    seq_cn = segments.sequence_cn.to_numpy(float)
    nseg, nv, nj = len(segments), 2*len(segments), len(sv)
    ref_pairs = [(i, i + 1) for i in range(nseg - 1)
                 if segments.chrom.iloc[i] == segments.chrom.iloc[i + 1]
                 and int(ends[i]) == int(starts[i + 1])]
    nr = len(ref_pairs)

    # Variables are reference CN, junction CN, endpoint source and latent
    # parent-CBS CN. Every graph cut from one parent shares the same CN.
    off_ref, off_j = 0, nr
    off_recip, off_src = nr + nj, nr + 3 * nj
    off_cn = off_src + nv
    incident = [[] for _ in range(nv)]
    for edge_index, (left, right) in enumerate(ref_pairs):
        incident[2*left+1].append(off_ref+edge_index)
        incident[2*right].append(off_ref+edge_index)
    edge_vertices = []
    junction_targets = []
    reciprocal_incident = [[] for _ in range(nv)]
    def cn_jump_at(position):
        left = np.flatnonzero(ends <= int(position))
        right = np.flatnonzero(starts >= int(position))
        if not len(left) or not len(right): return 0.0
        return abs(seq_cn[int(left[-1])] - seq_cn[int(right[0])])
    for j, row in sv.iterrows():
        if row.chrom1 == a.chrom:
            v1,b1,s1 = endpoint_vertex(row.pos1,row.orientation[0],starts,ends)
            chr_pos, external_chrom, external_pos = int(row.pos1), row.chrom2, int(row.pos2)
        else:
            v1,b1,s1 = endpoint_vertex(row.pos2,row.orientation[1],starts,ends)
            chr_pos, external_chrom, external_pos = int(row.pos2), row.chrom1, int(row.pos1)
        incident[v1].append(off_j+j)
        if not row.is_external:
            v2,b2,s2 = endpoint_vertex(row.pos2,row.orientation[1],starts,ends)
            incident[v2].append(off_j+j)
            capacity = min(seq_cn[b1], seq_cn[b2])
            target = min(max(cn_jump_at(row.pos1), cn_jump_at(row.pos2)), capacity)
        else:
            v2,b2,s2 = -1,-1,"EXT"
            capacity = seq_cn[b1]
            target = min(cn_jump_at(chr_pos), capacity)
        edge_vertices.append((v1,v2,b1,b2,s1,s2,external_chrom,external_pos))
        junction_targets.append(target)

        # A detected adjacency leaves one reciprocal side at each breakend
        # whose partner may be absent from the candidate graph. Model this
        # unknown continuation with exactly the called JCN. It therefore
        # vanishes when JCN=0 and cannot give unused candidates free slack.
        for endpoint_index, vertex in enumerate((v1, v2)):
            if vertex < 0:
                continue
            opposite = vertex - 1 if vertex % 2 == 0 else vertex + 1
            if 0 <= opposite < nv:
                reciprocal_incident[opposite].append(
                    off_recip + 2 * j + endpoint_index)

    # A paired intra-chromosomal breakend is an adjacency, not a molecule end.
    # Source is hard-zero at ordinary and paired-breakend sides. It is allowed
    # only at telomeres and at CN steps whose missing topology is unresolved.
    breakend_vertices = {vertex for edge in edge_vertices for vertex in edge[:2]
                         if vertex >= 0}
    cn_step_vertices = set()
    for i, (left, right) in enumerate(ref_pairs):
        # Continuous CNS values need residual mass at every genuine dosage
        # discontinuity.  cnv_snap_min_jump is only an SV-matching evidence
        # threshold and must not decide flow feasibility.
        if abs(seq_cn[right] - seq_cn[left]) >= 1e-6:
            cn_step_vertices.update((2 * left + 1, 2 * right))
    source_allowed = np.zeros(nv, dtype=bool)
    # A CN step is the explicit evidence that the supplied adjacency set does
    # not fully explain dosage. Permit a labelled latent endpoint there even
    # when another paired SV touches the same side; all non-CN-step paired
    # breakends remain hard-zero.
    source_allowed[list(cn_step_vertices)] = True
    component_end_vertices = {vertex for vertex in range(nv)
                              if not any(off_ref <= edge < off_j
                                         for edge in incident[vertex])}
    source_allowed[list(component_end_vertices)] = True
    foldback_latent_vertices=set()
    for j,row in sv.iterrows():
        for vertex in edge_vertices[j][:2]:
            if vertex < 0:
                continue
            opposite = vertex - 1 if vertex % 2 == 0 else vertex + 1
            if 0 <= opposite < nv:
                if bool(row.is_foldback):
                    foldback_latent_vertices.add(opposite)
    rr=[]; cc=[]; vv=[]; rhs=[]
    def add(coeffs,target):
        r=len(rhs)
        for c,v in coeffs: rr.append(r); cc.append(c); vv.append(v)
        rhs.append(float(target))
    fw=1.0
    for vertex in range(nv):
        coeffs=([(edge,fw) for edge in incident[vertex]]
                + [(edge,fw) for edge in reciprocal_incident[vertex]]
                + [(off_src+vertex,fw), (off_cn + segments.parent_cnv_segment
                    .astype("category").cat.codes.iloc[vertex//2], -fw)])
        add(coeffs, 0.0)
    parent_codes = segments.parent_cnv_segment.astype("category").cat.codes.to_numpy()
    parent_levels = list(segments.parent_cnv_segment.astype("category").cat.categories)
    nparent = len(parent_levels)
    design=sparse.csr_matrix((vv,(rr,cc)),shape=(len(rhs),off_cn+nparent))
    dense=design.toarray(); target_vector=np.asarray(rhs)
    ref_upper = np.asarray([min(seq_cn[left], seq_cn[right])
                            for left, right in ref_pairs])
    if a.reference_ploidy_cap is not None:
        ref_upper = np.minimum(ref_upper, a.reference_ploidy_cap)
    if a.breakpoint_reference_ploidy_cap is not None:
        breakpoint_boundary = np.isin(
            [starts[right] for _, right in ref_pairs], endpoint_values)
        ref_upper[breakpoint_boundary] = np.minimum(
            ref_upper[breakpoint_boundary], a.breakpoint_reference_ploidy_cap)
    if a.junction_counts is None:
        raise ValueError("--junction-counts is required by the shared-yield Poisson model")
    if a.one_copy_junction_pairs is None or a.one_copy_junction_pairs <= 0:
        raise ValueError("--one-copy-junction-pairs must be a fixed positive constant")
    counts = pd.read_csv(a.junction_counts, sep="\t")
    count_lookup = {
        (str(row.chrom1), int(row.pos1), str(row.chrom2), int(row.pos2)):
        (float(row.junction_pairs), float(row.background_pairs))
        for row in counts.itertuples()
    }
    selected_counts = [count_lookup.get(
        (str(row.chrom1), int(row.pos1), str(row.chrom2), int(row.pos2)),
        (0.0, 0.0)) for row in sv.iloc[:n_intra].itertuples()]
    sv_y = np.asarray([item[0] for item in selected_counts], float)
    sv_eta = np.maximum(np.asarray([item[1] for item in selected_counts], float), 1.0)
    beta_sv = float(a.one_copy_junction_pairs)
    bounds=[(0,None)]*(off_cn+nparent)
    for i in range(nr):
        if (a.reference_ploidy_cap is not None
                or a.breakpoint_reference_ploidy_cap is not None):
            bounds[off_ref+i] = (0, ref_upper[i])
    for j,(_,_,b1,b2,_,_,_,_) in enumerate(edge_vertices):
        bounds[off_j+j]=(0,None)
        for endpoint_index, vertex in enumerate(edge_vertices[j][:2]):
            reciprocal_index = off_recip + 2 * j + endpoint_index
            if vertex < 0:
                bounds[reciprocal_index] = (0, 0)
                continue
            opposite = vertex - 1 if vertex % 2 == 0 else vertex + 1
            bounds[reciprocal_index] = (0, None)
    for vertex in range(nv):
        if not source_allowed[vertex]:
            bounds[off_src + vertex] = (0, 0)
    # Shared-yield Poisson component:
    #   Y_e ~ Poisson(a * J_e + b_e)
    # with one shared a, hard side-flow conservation and non-negative flows.
    # CN-derived JCN targets and EagleC probabilities do not enter the data
    # likelihood. EagleC is only the candidate-edge generator.
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CVXPY is required; run this script in the graph-flow-cvxpy conda environment"
        ) from exc

    variable_count = off_cn + nparent
    cvx_x = cp.Variable(variable_count, nonneg=True)
    flow_matrix = dense[:nv] / fw
    flow_target = target_vector[:nv] / fw
    constraints = [flow_matrix @ cvx_x == flow_target]
    for index, (_, upper_bound) in enumerate(bounds):
        if upper_bound is not None and np.isfinite(upper_bound):
            constraints.append(cvx_x[index] <= float(upper_bound))

    # Edge flow cannot exceed the latent CN of either incident parent segment.
    for edge_index, (left, right) in enumerate(ref_pairs):
        constraints.extend([
            cvx_x[off_ref + edge_index] <= cvx_x[off_cn + parent_codes[left]],
            cvx_x[off_ref + edge_index] <= cvx_x[off_cn + parent_codes[right]],
        ])
    for j, (_, _, b1, b2, _, _, _, _) in enumerate(edge_vertices):
        constraints.append(cvx_x[off_j + j] <= cvx_x[off_cn + parent_codes[b1]])
        if b2 >= 0:
            constraints.append(
                cvx_x[off_j + j] <= cvx_x[off_cn + parent_codes[b2]])
        for endpoint_index, vertex in enumerate(edge_vertices[j][:2]):
            if vertex < 0:
                continue
            opposite = vertex - 1 if vertex % 2 == 0 else vertex + 1
            constraints.append(
                cvx_x[off_recip + 2*j + endpoint_index]
                <= cvx_x[off_cn + parent_codes[opposite // 2]])

    # A called adjacency can leave the reciprocal side of either breakpoint
    # unresolved when the companion derivative adjacency is absent from the
    # candidate graph.  Such continuation is edge-specific and may exist only
    # to the extent that the corresponding junction is used.  Consequently an
    # unused candidate (J_e = 0) cannot open free source capacity.
    for j in range(nj):
        for endpoint_index in range(2):
            reciprocal_index = off_recip + 2 * j + endpoint_index
            constraints.append(cvx_x[reciprocal_index] <= cvx_x[off_j + j])

    poisson_mean = sv_eta + beta_sv * cvx_x[off_j:off_j+n_intra]
    poisson_nll = cp.sum(
        poisson_mean - cp.multiply(sv_y, cp.log(poisson_mean))
    )
    parent_observed = np.asarray([
        segments.loc[segments.parent_cnv_segment.eq(level), "sequence_cn"].iloc[0]
        for level in parent_levels], float)
    parent_probes = np.asarray([
        float(parent.loc[int(level.replace("CNSEG", "")) - 1, "probes"])
        if "probes" in parent.columns else 1.0 for level in parent_levels])
    parent_sigma = np.maximum(
        a.cn_sigma_floor, a.cn_sigma_scale / np.sqrt(np.maximum(parent_probes, 1)))
    cn_nll = cp.sum(cp.abs(
        (cvx_x[off_cn:off_cn+nparent] - parent_observed) / parent_sigma))
    data_nll = cn_nll + poisson_nll

    # Stage 1: obtain the best data fit with source available. Stage 2 then
    # finds the fewest loose-end locations inside a likelihood tolerance.
    stage1 = cp.Problem(cp.Minimize(data_nll), constraints)
    stage1.solve(solver="CLARABEL", verbose=False)
    if stage1.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"stage-1 latent-CN/Poisson fit failed: {stage1.status}")
    minimum_data_nll = float(stage1.value)
    stage1_source = np.asarray(cvx_x.value[off_src:off_cn]).reshape(-1)
    allowed_vertices = np.asarray(sorted(
        {int(v) for v in np.flatnonzero(stage1_source > 1e-6)}
        - component_end_vertices), int)
    print(f"stage1 source-support candidates: {len(allowed_vertices)}", flush=True)
    base_sparse_constraints = list(constraints)
    base_sparse_constraints.append(
        data_nll <= minimum_data_nll + a.likelihood_tolerance)
    candidate_set = set(map(int, allowed_vertices))
    screened_out = (set(np.flatnonzero(source_allowed))
                    - candidate_set - component_end_vertices)
    fixed_zero = set(map(int, screened_out))
    # Deterministic lexicographic backward elimination. A location is removed
    # only when the exact Poisson+CN likelihood tolerance and hard flow remain
    # feasible. Repeat passes until the support is inclusion-minimal.
    while True:
        removed = 0
        order = sorted(candidate_set,
                       key=lambda v: (stage1_source[v], v))
        for vertex in order:
            trial_zero = fixed_zero | {vertex}
            trial_constraints = (base_sparse_constraints
                + [cvx_x[off_src + v] == 0 for v in sorted(trial_zero)])
            trial = cp.Problem(cp.Minimize(cp.sum(cvx_x[off_src:off_cn])),
                               trial_constraints)
            try:
                trial.solve(solver="CLARABEL", verbose=False)
            except cp.error.SolverError:
                continue
            trial_x = (None if cvx_x.value is None else
                       np.asarray(cvx_x.value).reshape(-1))
            likelihood_ok = (data_nll.value is not None
                and np.isfinite(data_nll.value)
                and data_nll.value <= minimum_data_nll
                    + a.likelihood_tolerance + 1e-4)
            flow_ok = (trial_x is not None and np.max(np.abs(
                flow_matrix @ trial_x - flow_target)) <= 1e-5)
            if (trial.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and likelihood_ok and flow_ok):
                fixed_zero.add(vertex)
                candidate_set.remove(vertex)
                removed += 1
        print(f"source backward-elimination pass: retained={len(candidate_set)} "
              f"removed={removed}", flush=True)
        if removed == 0:
            break
    final_constraints = (base_sparse_constraints
        + [cvx_x[off_src + v] == 0 for v in sorted(fixed_zero)])
    stage3 = cp.Problem(cp.Minimize(cp.sum(cvx_x[off_src:off_cn])),
                        final_constraints)
    stage3.solve(solver="CLARABEL", verbose=False)
    problem = stage3
    solver = "CLARABEL"
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"convex Poisson hard-flow balance failed: {problem.status}")

    class ConvexResult:
        pass
    joint = ConvexResult()
    joint.x = np.asarray(cvx_x.value, dtype=float).reshape(-1)
    joint.success = True
    joint.message = f"CVXPY lexicographic {solver}: {problem.status}"
    joint.fun = float(problem.value)

    if not joint.success: raise RuntimeError("joint Poisson balance failed: "+joint.message)
    x=joint.x
    reference_cn=x[:off_j]
    junction_cn=x[off_j:off_recip]
    reciprocal_cn=x[off_recip:off_src]
    source_cn=x[off_src:off_cn]
    latent_parent_cn=x[off_cn:off_cn+nparent]
    seq_cn = latent_parent_cn[parent_codes]
    flow_residual=design[:nv].dot(x)/fw
    reciprocal_source_cn = np.zeros(nv, float)
    for vertex, edges in enumerate(reciprocal_incident):
        reciprocal_source_cn[vertex] = sum(
            reciprocal_cn[edge - off_recip] for edge in edges)
    total_source_cn = source_cn + reciprocal_source_cn
    reciprocal_latent_vertices = {
        vertex for vertex, edges in enumerate(reciprocal_incident) if edges}

    segments["observed_sequence_cn"] = segments.sequence_cn
    segments["sequence_cn"] = seq_cn
    segments["balanced_sequence_cn"] = seq_cn
    segments.to_csv(a.outdir/"F1_chr18.balanced_sequence_cn.tsv",sep="\t",index=False)
    pd.DataFrame({
        "parent_cnv_segment": parent_levels,
        "observed_cn": parent_observed,
        "latent_cn": latent_parent_cn,
        "sigma_cn": parent_sigma,
        "standardized_residual": (latent_parent_cn-parent_observed)/parent_sigma,
    }).to_csv(a.outdir/"F1_chr18.latent_parent_cn.tsv", sep="\t", index=False)
    ref_boundaries = np.asarray([starts[right] for _, right in ref_pairs])
    pd.DataFrame({"edge_id":[f"REF{i+1:03d}" for i in range(nr)],"chrom":a.chrom,
        "boundary":ref_boundaries,"reference_cn":reference_cn}).to_csv(
        a.outdir/"F1_chr18.balanced_reference_cn.tsv",sep="\t",index=False)
    out=[]
    for j,row in sv.iterrows():
        _,_,b1,b2,s1,s2,external_chrom,external_pos=edge_vertices[j]
        cap=seq_cn[b1] if b2 < 0 else min(seq_cn[b1],seq_cn[b2])
        observed_pairs = sv_y[j] if j < n_intra else np.nan
        background_eta = sv_eta[j] if j < n_intra else np.nan
        out.append({"sv_id":row.sv_id,"chrom1":row.chrom1,"pos1":int(row.pos1),"chrom2":row.chrom2,
            "pos2":int(row.pos2),"orientation":row.orientation,"eaglec2_probability":row.eaglec2_probability,
            "is_external":bool(row.is_external),"is_foldback":bool(row.is_foldback),
            "vertex1":f"{a.chrom}:{int(row.pos1 if row.chrom1 == a.chrom else row.pos2)}:{s1}",
            "vertex2":f"{external_chrom}:{external_pos}:{s2}" if row.is_external else f"{a.chrom}:{int(row.pos2)}:{s2}",
            "junction_cn":junction_cn[j],"endpoint_cn_capacity":cap,
            "reciprocal_source1_cn":reciprocal_cn[2*j],
            "reciprocal_source2_cn":reciprocal_cn[2*j+1],
            "observed_cn_jump_diagnostic":junction_targets[j],
            "junction_pairs":observed_pairs,"junction_window_bp":50000,"poisson_background_eta":background_eta,
            "sv_reads_per_cn":beta_sv if j < n_intra else np.nan,
            "poisson_expected_pairs":background_eta+beta_sv*junction_cn[j] if j < n_intra else np.nan,
            "junction_fraction_of_capacity":junction_cn[j]/cap if cap>0 else np.nan,
            "cnv_identifiable":bool(junction_cn[j]>1e-3 and cap>0)})
    junction=pd.DataFrame(out)
    junction.to_csv(a.outdir/"F1_chr18.balanced_junction_cn.tsv",sep="\t",index=False)
    pd.DataFrame({"segment_id":np.repeat(segments.segment_id,2),"chrom":a.chrom,
        "side":np.tile(["L","R"],nseg),"source_cn":total_source_cn,
        "generic_source_cn":source_cn,
        "reciprocal_source_cn":reciprocal_source_cn}).to_csv(
        a.outdir/"F1_chr18.source_slack_cn.tsv",sep="\t",index=False)

    side_connected = np.maximum(np.repeat(seq_cn, 2) - total_source_cn, 0)
    side_qc = pd.DataFrame({
        "segment_id": np.repeat(segments.segment_id, 2),
        "chrom": a.chrom,
        "side": np.tile(["L", "R"], nseg),
        "sequence_cn": np.repeat(seq_cn, 2),
        "connected_cn": side_connected,
        "source_cn": total_source_cn,
        "generic_source_cn": source_cn,
        "reciprocal_source_cn": reciprocal_source_cn,
        "connected_fraction": np.divide(
            side_connected, np.repeat(seq_cn, 2),
            out=np.zeros(nv), where=np.repeat(seq_cn, 2) > 0),
        "source_allowed": source_allowed,
        "is_unresolved_cn_step": [v in cn_step_vertices for v in range(nv)],
        "is_sv_breakend": [v in breakend_vertices for v in range(nv)],
        "is_reciprocal_latent_side": [v in reciprocal_latent_vertices
                                       for v in range(nv)],
        "is_telomere": [v in component_end_vertices for v in range(nv)],
    })
    side_qc.to_csv(a.outdir/"F1_chr18.connected_copy_fraction.tsv",
                   sep="\t", index=False)
    boundary_capacity = np.asarray([min(seq_cn[left], seq_cn[right])
                                    for left, right in ref_pairs])
    boundary_qc = pd.DataFrame({
        "edge_id": [f"REF{i+1:03d}" for i in range(nr)],
        "chrom": a.chrom,
        "boundary": ref_boundaries,
        "reference_cn": reference_cn,
        "boundary_capacity_cn": boundary_capacity,
        "reference_continuity_ratio": np.divide(
            reference_cn, boundary_capacity, out=np.zeros(nr),
            where=boundary_capacity > 0),
        "is_sv_breakpoint": np.isin(ref_boundaries, endpoint_values),
    })
    boundary_qc.to_csv(a.outdir/"F1_chr18.reference_continuity_qc.tsv",
                       sep="\t", index=False)

    fig_qc, (qs, qr) = plt.subplots(2, 1, figsize=(15, 6), sharex=False)
    ordinary_side = ~(side_qc.is_sv_breakend
                      | side_qc.is_reciprocal_latent_side
                      | side_qc.is_telomere)
    qs.hist(side_qc.loc[ordinary_side, "connected_fraction"], bins=30,
            range=(0, 1.05), color="#3182bd")
    qs.axvline(1, color="black", ls="--", lw=1)
    qs.set(xlabel="connected copy fraction", ylabel="ordinary segment sides",
           title="Reference-default graph balance QC")
    ordinary_boundary = ~boundary_qc.is_sv_breakpoint
    qr.scatter(boundary_qc.loc[ordinary_boundary, "boundary"] / 1e6,
               boundary_qc.loc[ordinary_boundary, "reference_continuity_ratio"],
               s=18, color="#238b45", label="ordinary boundary")
    qr.scatter(boundary_qc.loc[~ordinary_boundary, "boundary"] / 1e6,
               boundary_qc.loc[~ordinary_boundary, "reference_continuity_ratio"],
               s=24, color="#cb181d", label="SV breakpoint")
    qr.axhline(1, color="black", ls="--", lw=1)
    qr.set(xlabel=f"{a.chrom} boundary (Mb)", ylabel="reference CN / boundary capacity")
    qr.legend(frameon=False)
    fig_qc.tight_layout()
    fig_qc.savefig(a.outdir/"F1_chr18.graph_connectivity_qc.png", dpi=220)
    plt.close(fig_qc)

    fig,(ax,aj)=plt.subplots(2,1,figsize=(16,7),sharex=True,gridspec_kw={"height_ratios":[2.1,1]})
    bins=pd.read_csv(a.cnr,sep="\t",comment="#")
    bins=bins[bins.chromosome.astype(str).eq(a.chrom)].copy()
    bins["copy_number"]=a.ploidy*np.exp2(pd.to_numeric(bins.log2,errors="coerce"))
    centers=(bins.start.to_numpy(float)+bins.end.to_numpy(float))/(2e6)
    parent["observed_copy_number"] = parent.copy_number
    for parent_index, level in enumerate(parent_levels):
        original_index = int(level.replace("CNSEG", "")) - 1
        parent.loc[original_index, "copy_number"] = latent_parent_cn[parent_index]
    normal_cn=bins.copy_number.replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
    ymax=max(3.0,float(np.nanpercentile(normal_cn, 99.5))*1.08,
             float(parent.copy_number.max())*1.15)
    visible=bins.copy_number.le(ymax)
    bin_size_kb = int(round(np.median(bins.end.to_numpy(float) -
                                      bins.start.to_numpy(float)) / 1000))
    ax.scatter(centers[visible],bins.loc[visible,"copy_number"],s=7,color="#9ecae1",alpha=.5,
               linewidths=0,label=f"{bin_size_kb}-kb CNV bins (pre-CBS)",zorder=1)
    # One staircase represents contiguous [start, end) CBS intervals exactly;
    # adjacent segments share a boundary but never overlap in genomic span.
    edges=np.r_[parent.start.iloc[0],parent.end.to_numpy(float)]/1e6
    stair_values = np.r_[parent.copy_number.to_numpy(float),
                         parent.copy_number.iloc[-1]]
    ax.step(edges, stair_values, where="post", color="#08519c", lw=2.2,
            label="CBS segments [start, end)", zorder=3)
    segment_worst_connected = 1 - np.divide(
        total_source_cn.reshape(nseg, 2).max(axis=1), seq_cn,
        out=np.zeros(nseg), where=seq_cn > 0)
    label_offset = max(ymax * .018, .08)
    for i, row in enumerate(segments.itertuples()):
        center = (row.start + row.end) / 2e6
        poor = segment_worst_connected[i] < .10 and not (
            i == 0 or i == nseg - 1)
        ax.text(
            center, min(row.sequence_cn + label_offset, ymax * .96),
            row.segment_id, rotation=90, ha="center", va="bottom",
            fontsize=5.8, color="#b2182b" if poor else "0.25",
            fontweight="bold" if poor else "normal", zorder=5,
            bbox=({"facecolor": "#fee0d2", "edgecolor": "none", "alpha": .75,
                   "pad": .7} if poor else None),
        )
    ax.set_ylim(-.1,ymax)
    ax.legend(loc="upper right",frameon=False)
    ax.set_ylabel("continuous copy number"); ax.set_title(
        f"F1 {a.chrom}: pre-CBS CNV bins, graph segments, and CNV-balanced EagleC2 junction CN")
    for row in junction.itertuples():
        left,right,height=row.pos1/1e6,row.pos2/1e6,row.junction_cn
        color="#b2182b" if row.cnv_identifiable else "0.65"
        aj.plot([left,left,right,right],[0,height,height,0],color=color,lw=1.6)
        aj.text((left+right)/2,height,row.sv_id,ha="center",va="bottom",fontsize=7)
    aj.set(xlabel=f"{a.chrom} position (Mb)",ylabel="junction CN")
    fig.tight_layout(); fig.savefig(a.outdir/"F1_chr18.cnv_jcn_balance.png",dpi=220); plt.close(fig)
    with (a.outdir/"F1_chr18.balance_summary.txt").open("w") as f:
        f.write("sequence_cn_source\tCNVkit CBS .cns\nsequence_cn_conversion\tCN=ploidy*2^log2\n")
        f.write(f"nominal_ploidy\t{a.ploidy}\nreference_ploidy_cap\t{a.reference_ploidy_cap}\nbreakpoint_reference_ploidy_cap\t{a.breakpoint_reference_ploidy_cap}\nsequence_cn_optimization\tlatent_parent_CBS_CN\nsv_read_likelihood\tPoisson\n")
        f.write(f"chromosome\t{a.chrom}\nparent_cnv_segments\t{len(parent)}\ngraph_sequence_segments\t{nseg}\nsv_edges\t{nj}\nintrachrom_sv_edges\t{n_intra}\ninterchrom_sv_edges\t{nj-n_intra}\n")
        f.write(f"cnv_snap_tolerance_bp\t{a.cnv_snap_tolerance}\ncnv_snap_min_jump\t{a.cnv_snap_min_jump}\ncnv_snapped_boundaries\t{len(snap_rows)}\n")
        f.write("optimization_model\tlexicographic_latent_CN_Poisson_hard_flow_sparse_source\n")
        f.write("objective\tstage1_CN_Laplace_plus_Poisson;stage2_min_source_locations;stage3_min_source_mass\n")
        f.write("flow_conservation\thard_equality\n")
        f.write("eaglec_probability_role\tcandidate_selection_only\n")
        f.write("cn_junction_target\tdisabled\n")
        f.write("reciprocal_continuation\tSV_tied_0_le_U_le_JCN\n")
        f.write(f"likelihood_tolerance\t{a.likelihood_tolerance}\n")
        f.write("source_penalty_form\tlocation_sparsity_then_mass\n")
        f.write("source_location_solver\tlikelihood-constrained_backward_elimination\n")
        f.write("ordinary_internal_source\thard_zero\nnon_cn_step_paired_breakend_source\thard_zero\ncn_step_latent_endpoint\tallowed_sparse\n")
        f.write(f"junction_window_bp\t50000\nshared_junction_yield_a\t{beta_sv}\nbackground_model\tlocal_background_pairs\n")
        f.write(f"solver_status\t{joint.message}\nstage1_minimum_data_nll\t{minimum_data_nll:.10g}\nobjective_value\t{joint.fun:.10g}\nmax_flow_residual_cn\t{np.max(np.abs(flow_residual)):.10g}\n")
        active_unresolved = sum(source_cn[v] > 1e-6 for v in range(nv)
                                if v not in component_end_vertices)
        f.write(f"active_unresolved_source_locations\t{active_unresolved}\n")
        f.write(f"max_abs_latent_cn_shift\t{np.max(np.abs(latent_parent_cn-parent_observed)):.10g}\n")
        f.write(f"total_junction_cn\t{junction_cn.sum():.10g}\ntotal_source_cn\t{total_source_cn.sum():.10g}\n")
        f.write(f"total_generic_source_cn\t{source_cn.sum():.10g}\n")
        f.write(f"total_reciprocal_source_cn\t{reciprocal_source_cn.sum():.10g}\n")
        f.write(f"ordinary_side_median_connected_fraction\t{side_qc.loc[ordinary_side, 'connected_fraction'].median():.10g}\n")
        f.write(f"ordinary_boundary_median_reference_ratio\t{boundary_qc.loc[ordinary_boundary, 'reference_continuity_ratio'].median():.10g}\n")
    print(a.outdir)

if __name__ == "__main__": main()
