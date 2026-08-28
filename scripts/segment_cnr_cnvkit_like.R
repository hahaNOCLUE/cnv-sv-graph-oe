#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(DNAcopy))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop("usage: segment_cnr_cnvkit_like.R INPUT.cnr OUTPUT.cns OUTPUT.cbs.tsv [alpha] [undo.SD]")
}

input <- args[[1]]
output_cns <- args[[2]]
output_cbs <- args[[3]]
alpha <- if (length(args) >= 4L) as.numeric(args[[4]]) else 1e-4
undo_sd <- if (length(args) >= 5L) as.numeric(args[[5]]) else 1

bins <- read.delim(input, check.names = FALSE, stringsAsFactors = FALSE)
required <- c("chromosome", "start", "end", "log2")
if (!all(required %in% names(bins))) {
  stop("input CNR is missing columns: ", paste(setdiff(required, names(bins)), collapse = ", "))
}

bins$log2 <- as.numeric(bins$log2)
bins$weight <- if ("weight" %in% names(bins)) as.numeric(bins$weight) else 1
bins$depth <- if ("depth" %in% names(bins)) as.numeric(bins$depth) else 2^bins$log2
usable <- is.finite(bins$log2) & is.finite(bins$weight) & bins$weight > 0
work <- bins[usable, , drop = FALSE]
if (!nrow(work)) stop("no usable bins")

midpoint <- floor((work$start + work$end) / 2) + 1
cna <- CNA(work$log2, work$chromosome, midpoint,
           data.type = "logratio", sampleid = "F1_10kb.continuous")
smoothed <- smooth.CNA(cna, outlier.SD.scale = 3, smooth.SD.scale = 2)
fit <- segment(smoothed, alpha = alpha, min.width = 2,
               undo.splits = "sdundo", undo.SD = undo_sd,
               verbose = 1)
seg <- fit$output

chrom_order <- unique(work$chromosome)
rows <- vector("list", nrow(seg))
for (i in seq_len(nrow(seg))) {
  chrom <- as.character(seg$chrom[i])
  local <- work$chromosome == chrom &
    midpoint >= seg$loc.start[i] & midpoint <= seg$loc.end[i]
  selected <- work[local, , drop = FALSE]
  if (!nrow(selected)) stop("CBS segment has no matching bins: ", chrom, ":", seg$loc.start[i])
  w <- selected$weight
  rows[[i]] <- data.frame(
    chromosome = chrom,
    start = min(selected$start),
    end = max(selected$end),
    gene = "-",
    log2 = as.numeric(seg$seg.mean[i]),
    depth = weighted.mean(selected$depth, w),
    probes = nrow(selected),
    weight = sum(w),
    stringsAsFactors = FALSE
  )
}

cns <- do.call(rbind, rows)
cns$chromosome <- factor(cns$chromosome, levels = chrom_order)
cns <- cns[order(cns$chromosome, cns$start), ]
cns$chromosome <- as.character(cns$chromosome)

write.table(cns, output_cns, sep = "\t", quote = FALSE, row.names = FALSE)
write.table(seg, output_cbs, sep = "\t", quote = FALSE, row.names = FALSE)
