#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(gGnome)
  library(GenomicRanges)
  library(data.table)
})

# This JaBbA environment contains gGnome built with its src.blank CPLEX stub.
# Supply the same small MILP interface through lpSolve while retaining the
# official gGnome maxflow/peel implementation.
Rcplex2_lpsolve <- function(cvec, Amat, bvec, Qmat = NULL, lb = 0, ub = Inf,
                            control = list(), objsense = "min", sense = "L",
                            vtype = NULL, n = 1, tuning = FALSE) {
  if (!is.null(Qmat)) stop("lpSolve adapter supports the linear peel MILP only")
  A <- as.matrix(Amat)
  nc <- ncol(A)
  if (length(lb) == 1L) lb <- rep(lb, nc)
  if (length(ub) == 1L) ub <- rep(ub, nc)
  finite_ub <- which(is.finite(ub))
  positive_lb <- which(is.finite(lb) & lb > 0)
  if (length(finite_ub)) {
    U <- matrix(0, length(finite_ub), nc)
    U[cbind(seq_along(finite_ub), finite_ub)] <- 1
    A <- rbind(A, U); bvec <- c(bvec, ub[finite_ub]); sense <- c(sense, rep("L", length(finite_ub)))
  }
  if (length(positive_lb)) {
    L <- matrix(0, length(positive_lb), nc)
    L[cbind(seq_along(positive_lb), positive_lb)] <- 1
    A <- rbind(A, L); bvec <- c(bvec, lb[positive_lb]); sense <- c(sense, rep("G", length(positive_lb)))
  }
  fit <- lpSolve::lp(direction = objsense, objective.in = cvec,
                     const.mat = A, const.dir = c(L = "<=", G = ">=", E = "=")[sense],
                     const.rhs = bvec, all.int = !is.null(vtype) && all(vtype == "I"),
                     scale = 0)
  if (fit$status != 0) stop("lpSolve peel MILP failed with status ", fit$status)
  list(xopt = fit$solution, obj = fit$objval, status = fit$status,
       extra = list(nodecnt = NA_real_, slack = NA_real_), epgap = 0)
}
gns <- asNamespace("gGnome")
unlockBinding("Rcplex2", gns)
assign("Rcplex2", Rcplex2_lpsolve, envir = gns)
lockBinding("Rcplex2", gns)

root <- "/home/dell/a1/microc"
args <- commandArgs(trailingOnly = TRUE)
indir <- if (length(args)) normalizePath(args[[1]], mustWork = TRUE) else
  file.path(root, "result_2/compartment/chr18_cnv_jcn_balance")
scale_cn <- 100

seg <- fread(file.path(indir, "F1_chr18.balanced_sequence_cn.tsv"))
ref <- fread(file.path(indir, "F1_chr18.balanced_reference_cn.tsv"))
sv <- fread(file.path(indir, "F1_chr18.balanced_junction_cn.tsv"))

nodes <- GRanges(seg$chrom, IRanges(seg$start + 1L, seg$end))
mcols(nodes)$segment_id <- seg$segment_id
seqlengths(nodes)["chr18"] <- max(seg$end)

ref_edges <- data.table(
  n1 = seq_len(nrow(seg) - 1L), n1.side = 1L,
  n2 = seq_len(nrow(seg) - 1L) + 1L, n2.side = 0L,
  type = "REF", label = ref$edge_id,
  cn = ref$reference_cn * scale_cn
)

vertex_node <- function(vertex, pos) {
  side <- sub("^.*:", "", vertex)
  if (side == "L") which(seg$start == pos)[1] else which(seg$end == pos)[1]
}
alt_edges <- rbindlist(lapply(seq_len(nrow(sv)), function(i) {
  data.table(
    n1 = vertex_node(sv$vertex1[i], sv$pos1[i]),
    n1.side = as.integer(endsWith(sv$vertex1[i], ":R")),
    n2 = vertex_node(sv$vertex2[i], sv$pos2[i]),
    n2.side = as.integer(endsWith(sv$vertex2[i], ":R")),
    type = "ALT", label = sv$sv_id[i], cn = sv$junction_cn[i] * scale_cn
  )
}))
edges <- rbind(ref_edges, alt_edges, fill = TRUE)
if (anyNA(edges[, .(n1, n2, n1.side, n2.side, cn)])) stop("Unresolved graph endpoint")
# peel subtracts integer walk multiplicities.  Quantize the uniformly scaled
# graph, so every subsequent subtraction preserves exact node-edge balance.
edges[, cn := round(cn)]

# gGnome requires edge flow at either side of a node to be no greater than
# node CN.  Preserve REF/JCN and make the smallest possible node-CN projection;
# source CN remains the exact residual (node CN - incident flow) at each side.
side_flow <- matrix(0, nrow(seg), 2, dimnames = list(seg$segment_id, c("L", "R")))
for (i in seq_len(nrow(edges))) {
  side_flow[edges$n1[i], edges$n1.side[i] + 1L] <-
    side_flow[edges$n1[i], edges$n1.side[i] + 1L] + edges$cn[i]
  side_flow[edges$n2[i], edges$n2.side[i] + 1L] <-
    side_flow[edges$n2[i], edges$n2.side[i] + 1L] + edges$cn[i]
}
target_node_cn <- round(seg$balanced_sequence_cn * scale_cn)
projected_node_cn <- pmax(target_node_cn, side_flow[, "L"], side_flow[, "R"])
mcols(nodes)$cn <- projected_node_cn

source_before <- fread(file.path(indir, "F1_chr18.source_slack_cn.tsv"))
source_after <- data.table(
  segment_id = rep(seg$segment_id, each = 2L), chrom = "chr18",
  side = rep(c("L", "R"), nrow(seg)),
  source_cn_projected = as.vector(t((projected_node_cn - side_flow) / scale_cn))
)
source_compare <- merge(source_before, source_after,
                        by = c("segment_id", "chrom", "side"), all = TRUE)
source_compare[, delta_cn := source_cn_projected - source_cn]
fwrite(source_compare,
       file.path(indir, "F1_chr18.gGnome_source_cn_projection.tsv"), sep = "\t")
fwrite(data.table(segment_id = seg$segment_id,
                  sequence_cn_before = target_node_cn / scale_cn,
                  sequence_cn_projected = projected_node_cn / scale_cn,
                  delta_cn = (projected_node_cn - target_node_cn) / scale_cn),
       file.path(indir, "F1_chr18.gGnome_node_cn_projection.tsv"), sep = "\t")

gg <- gG(nodes = nodes, edges = edges,
         meta = list(name = "F1 chr18 CNV/JCN graph", y.field = "cn"))
saveRDS(gg, file.path(indir, "F1_chr18.gGnome_graph.scaled100.rds"))

cache_file <- file.path(indir, "F1_chr18.gGnome_peel.cache.rds")
export_cache <- length(args) >= 2L && args[[2]] == "--export-cache"
walks <- if (export_cache) {
  readRDS(cache_file)
} else {
  tryCatch(
    peel(gg, field = NULL, embed.loops = FALSE, verbose = TRUE,
         cache.path = cache_file),
    error = function(e) {
      if (!file.exists(cache_file)) stop(e)
      message("peel stopped with a small residual flow; exporting cached walks: ",
              conditionMessage(e))
      readRDS(cache_file)
    }
  )
}
saveRDS(walks, file.path(indir, "F1_chr18.gGnome_peel_walks.rds"))

walk_dt <- copy(walks$dt)
walk_dt[, walk_cn := cn / scale_cn]
walk_dt[, node_count := vapply(walk_dt[["snode.id"]], base::length, integer(1))]
walk_dt[, edge_count := vapply(walk_dt[["sedge.id"]], base::length, integer(1))]
fwrite(walk_dt[, .(walk_id = walk.id, walk_cn, circular, length,
                   node_count, edge_count)],
       file.path(indir, "F1_chr18.gGnome_peel_walk_summary.tsv"), sep = "\t")

graph_nodes <- copy(walks$graph$nodes$dt)
node_rows <- rbindlist(lapply(seq_len(nrow(walk_dt)), function(i) {
  signed_ids <- unlist(walk_dt$snode.id[[i]])
  if (!length(signed_ids)) return(NULL)
  z <- graph_nodes[match(abs(signed_ids), node.id)]
  data.table(walk_id = walk_dt$walk.id[i], walk_cn = walk_dt$walk_cn[i],
             circular = walk_dt$circular[i], order = seq_along(signed_ids),
             node_id = abs(signed_ids), chrom = as.character(z$seqnames),
             start = z$start - 1L, end = z$end,
             strand = ifelse(signed_ids > 0, "+", "-"))
}), fill = TRUE)
fwrite(node_rows, file.path(indir, "F1_chr18.gGnome_peel_walk_nodes.tsv"), sep = "\t")

graph_edges <- copy(walks$graph$edges$dt)
edge_rows <- rbindlist(lapply(seq_len(nrow(walk_dt)), function(i) {
  signed_ids <- unlist(walk_dt$sedge.id[[i]])
  if (!length(signed_ids)) return(NULL)
  z <- graph_edges[match(abs(signed_ids), sedge.id)]
  data.table(walk_id = walk_dt$walk.id[i], walk_cn = walk_dt$walk_cn[i],
             order = seq_along(signed_ids), edge_id = abs(signed_ids),
             type = z$type, label = z$label, edge_cn = z$cn / scale_cn)
}), fill = TRUE)
fwrite(edge_rows, file.path(indir, "F1_chr18.gGnome_peel_walk_edges.tsv"), sep = "\t")

cat("walks", nrow(walk_dt), "\n")
cat("circular", sum(walk_dt$circular), "\n")

