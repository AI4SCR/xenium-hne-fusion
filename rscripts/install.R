#!/usr/bin/env Rscript
#
# Install R dependencies for rscripts/batch_correct.R (ADTnorm) into a
# project-local library, so no admin/system R library access is needed.
#
# Usage (from repo root):
#   module load r-light/4.5.2
#   Rscript rscripts/install.R

lib <- file.path(getwd(), "rscripts", ".Rlibs")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(lib)

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = lib)
}

# ADTnorm's Imports: flowCore/flowStats/EMDomics (Bioconductor), fda (CRAN).
BiocManager::install(
  c("flowCore", "flowStats", "EMDomics"),
  lib = lib, update = FALSE, ask = FALSE
)

install.packages(
  c("fda", "arrow", "remotes"),
  repos = "https://cloud.r-project.org", lib = lib
)

remotes::install_github("yezhengSTAT/ADTnorm", lib = lib, upgrade = "never")

cat("Done. R library:", lib, "\n")
