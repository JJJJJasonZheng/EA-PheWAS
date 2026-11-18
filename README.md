## EA-PheWAS: Integrating Phenotype Embeddings with PheWAS for Enhanced Gene-Phenotype Discovery

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.04.21.720031-blue.svg)](https://doi.org/10.64898/2026.04.21.720031)

**EA-PheWAS** (Embedding-Augmented PheWAS) combines conventional regression-based PheWAS with phenotype-embedding similarity scans for gene-level phenotype prioritization.

## Overview

1. **EmbedPheScan:** Computes mean-centered cosine similarities between gene carrier embeddings and phecode embeddings, using non-carrier bootstrap resampling (`sample_from="noncarriers"`) to calculate *z*-score p-values.
2. **Conventional PheWAS:** Fits Firth penalized logistic regression adjusted for age, sex, and `PC1`–`PC10`, with Fisher's exact test as a fallback when needed.
3. **ACAT Aggregation:** Combines p-values from EmbedPheScan and conventional PheWAS with the Aggregated Cauchy Association Test (ACAT).

## Repository Structure

```text
code/
├── EA_PheWAS.py             # Main script for running a single-gene scan
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

Run on the example data:

```bash
python code/EA_PheWAS.py --gene PKD1 --out_dir ./results
```

If `--icd_embed` and `--ind_embed` are not provided, `EA_PheWAS.py` trains ICD-10 embeddings from `--hesin` via `embed_w2v.py` and builds `loaded_embeddings`, `phecode_embeddings`, and `phecode_mapping` using the helpers in `code/ACAT_PheWAS.py`. To run on custom data, pass `--hesin`, `--cov`, `--carrier_file`, and optional `--icd_embed` / `--ind_embed` / `--icd_freq` paths (see `python code/EA_PheWAS.py -h`).
