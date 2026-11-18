## EA-PheWAS: Integrating Phenotype Embeddings with PheWAS for Enhanced Gene-Phenotype Discovery

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.04.21.720031-blue.svg)](https://doi.org/10.64898/2026.04.21.720031)

**EA-PheWAS** (Embedding-Augmented PheWAS) combines conventional regression-based PheWAS with phenotype-embedding similarity scans for gene-level phenotype prioritization.

## Overview

1. **EmbedPheScan:** Computes cosine similarities between gene carrier embeddings and phecode embeddings, using randomly sampled pseudo-carrier groups to calculate *z*-score p-values.
2. **Conventional PheWAS:** Fits covariate-adjusted Firth logistic regression, with Fisher's exact test as a fallback when needed.
3. **ACAT Aggregation:** Combines p-values from EmbedPheScan and conventional PheWAS using the Aggregated Cauchy Association Test (ACAT).

## Repository Structure

```text
code/
├── EA_PheWAS.py             # Main pipeline for running EA-PheWAS
├── EmbedPheScan.py          # Embedding-based scan
├── PheWAS.py                # Firth logistic regression & Fisher exact test
├── ACAT_PheWAS.py           # Embedding/phecode helpers and ACAT combination
└── eval_hit_precision.py    # Precision@K evaluation with HPO annotations

embedding_model/
└── embed_w2v.py             # Word2Vec training for ICD-10 embeddings

sample_data/
├── cov_adjust_sample.txt        # Example covariate table
├── hesin_sample.txt             # Example ICD-10 diagnosis table
├── carrier_sample.txt           # Example carrier ID list
├── icd2phecode_des.csv          # ICD-10 to PheCode 1.2 mapping
├── Phecode_map12_filtered.csv   # PheCode 1.2 descriptions
└── hpo-phecode1.2_links.tsv     # HPO to PheCode 1.2 mapping
```

## Usage

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `gensim`, `pyhpo`.

Run on the included example data:

```bash
python code/EA_PheWAS.py --gene PKD1 --out_dir ./results
```

### Running on Custom Data

`sample_data/` already provides the ICD-10, PheCode 1.2, and HPO mapping tables. To run the pipeline on your own cohort, prepare three tab/space-separated files matching `sample_data/`:

- `--hesin`: participant ICD-10 diagnoses with columns `eid`, `diag_icd10`
- `--cov`: participant covariates with columns `eid`, `age_recruit`, `sex`, and `PC1`–`PC10`
- `--carrier_file`: variant carrier IDs for the target gene with column `sample`

Then run:

```bash
python code/EA_PheWAS.py \
  --gene PKD1 \
  --hesin /path/to/hesin.txt \
  --cov /path/to/cov_adjust.txt \
  --carrier_file /path/to/carriers.txt \
  --min_cases 20 \
  --out_dir ./results
```

If precomputed embeddings are not passed via `--icd_embed` and `--ind_embed`, `EA_PheWAS.py` trains ICD-10 embeddings from `--hesin` using `embed_w2v.py` and constructs the individual embeddings, phecode embeddings, and phecode mapping automatically. Outputs (`<GENE>_summary.pkl`, `<GENE>_hpo.pkl`, and `<GENE>_eval.pkl`) are written under `<out_dir>/<GENE>/`.
