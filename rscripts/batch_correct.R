#!/usr/bin/env Rscript
#
# Batch-correct cell-level protein expression with ADTnorm.
#
# ADTnorm (https://github.com/yezhengSTAT/ADTnorm) normalizes each protein marker
# independently via landmark registration: it detects density peaks/valleys
# (negative/positive populations) per sample and aligns them across samples --
# a different approach from the joint-embedding methods (Harmony/ComBat/Scanorama)
# already compared in scripts/eval/batch_correction.py, and one purpose-built for
# antibody-derived-tag (ADT) / protein-panel data specifically.
#
# Reads raw whole-sample protein counts directly from
# 01_structured/<name>/<sample_id>/proteins.parquet (same source as
# scripts/eval/protein_histograms.py / batch_correction.py), treating
# `sample_id` as the batch variable. Self-contained: does not depend on the
# Python package, only on the R library installed by install.R.
#
# Usage (from repo root):
#   module load r-light/4.5.2
#   Rscript rscripts/batch_correct.R \
#       --data_dir /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/mesothelioma/xenium-hne-fusion-v4 \
#       --name owkin \
#       --sample_ids CH_C_518a_x2,CH_C_523a_x2,CH_C_525a_x2,CH_C_526a_x1,CH_C_527a_x2 \
#       --run_name c_cells \
#       --debug true

.libPaths(file.path(getwd(), "rscripts", ".Rlibs"))

suppressPackageStartupMessages({
  library(arrow)
  library(ADTnorm)
})

PROTEIN_PANEL <- c(
  "Beta-catenin", "CD11c", "CD138", "CD16", "CD163-1", "CD20", "CD31", "CD3E-1", "CD4-1", "CD45",
  "CD45RA", "CD45RO", "CD68-1", "CD8A-1", "E-Cadherin", "GranzymeB", "HLA-DR", "Ki-67", "LAG-3", "PCNA",
  "PD-1", "PD-L1", "PTEN-1", "PanCK", "VISTA", "Vimentin", "alphaSMA"
)

protein_base_name <- function(protein) sub("-\\d+$", "", protein)

# Marker naming (e.g. "PCNA" vs "PCNA-1") is inconsistent across sample batches;
# normalize by matching on the base name shared with the panel (mirrors
# xenium_hne_fusion.targets.protein_base_to_panel).
load_sample_proteins <- function(structured_dir, sample_id, proteins) {
  base_to_panel <- setNames(proteins, protein_base_name(proteins))
  path <- file.path(structured_dir, sample_id, "proteins.parquet")
  df <- as.data.frame(arrow::read_parquet(path))
  df$geometry <- NULL

  base_names <- protein_base_name(colnames(df))
  is_protein_col <- base_names %in% names(base_to_panel)
  colnames(df)[is_protein_col] <- base_to_panel[base_names[is_protein_col]]

  out <- df[, c("cell_id", proteins)]
  # cell_id is only unique within a sample (Xenium per-run barcodes collide
  # across samples), so make it globally unique before using it as a row key.
  out$cell_id <- paste(sample_id, out$cell_id, sep = "_")
  out$sample_id <- sample_id
  out
}

subsample_per_batch <- function(df, n, seed) {
  set.seed(seed)
  parts <- lapply(split(df, df$sample_id), function(g) {
    g[sample.int(nrow(g), min(nrow(g), n)), , drop = FALSE]
  })
  do.call(rbind, parts)
}

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    parsed[[key]] <- args[i + 1]
    i <- i + 2
  }
  parsed
}

main <- function(args) {
  data_dir <- args[["data_dir"]]
  name <- args[["name"]]
  sample_ids <- strsplit(args[["sample_ids"]], ",")[[1]]
  run_name <- args[["run_name"]]
  debug <- isTRUE(as.logical(args[["debug"]]))
  debug_cells_per_batch <- as.integer(if (is.null(args[["debug_cells_per_batch"]])) 500 else args[["debug_cells_per_batch"]])
  seed <- as.integer(if (is.null(args[["seed"]])) 0 else args[["seed"]])

  stopifnot(!is.null(data_dir), !is.null(name), !is.null(run_name), length(sample_ids) > 0)

  structured_dir <- file.path(data_dir, "01_structured", name)

  cat(sprintf("Loading proteins for %d samples: %s\n", length(sample_ids), paste(sample_ids, collapse = ", ")))
  cells <- do.call(rbind, lapply(sample_ids, load_sample_proteins, structured_dir = structured_dir, proteins = PROTEIN_PANEL))

  if (debug) {
    cells <- subsample_per_batch(cells, debug_cells_per_batch, seed)
  }
  cat(sprintf("Loaded %d cells\n", nrow(cells)))

  cell_x_adt <- as.matrix(cells[, PROTEIN_PANEL])
  rownames(cell_x_adt) <- cells$cell_id
  cell_x_feature <- data.frame(sample = cells$sample_id, row.names = cells$cell_id)

  out_dir <- file.path(data_dir, "03_output", name, "anndata", "batch_correction")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  cat("Running ADTnorm...\n")
  normalized <- ADTnorm(
    cell_x_adt = cell_x_adt,
    cell_x_feature = cell_x_feature,
    save_outpath = file.path(out_dir, paste0(run_name, "_adtnorm_figures")),
    study_name = run_name,
    save_fig = FALSE,
    verbose = TRUE
  )

  normalized$cell_id <- rownames(cell_x_adt)
  normalized$sample_id <- cells$sample_id

  out_path <- file.path(out_dir, paste0(run_name, if (debug) "_debug" else "", "_adtnorm.parquet"))
  arrow::write_parquet(normalized, out_path)
  cat(sprintf("Wrote normalized ADT matrix to %s\n", out_path))
}

main(parse_args(commandArgs(trailingOnly = TRUE)))
