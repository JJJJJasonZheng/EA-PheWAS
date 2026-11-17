## EA-PheWAS: Integrating Phenotype Embeddings with PheWAS for Enhanced Gene-Phenotype Discovery

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.04.21.720031-blue.svg)](https://doi.org/10.64898/2026.04.21.720031)

**EA-PheWAS** (Embedding-Augmented PheWAS) is a unified framework that integrates signals from regression-based PheWAS and embedding-similarity–based phenotype prioritization.

## Overview

The pipeline leverages a dual-branch architecture to maximize statistical power and robustness:
1. **EmbedPheScan:** Computes mean-centered cosine similarities between gene-carrier embeddings and phecode embeddings, calibrating normal-approximation *z*-score p-values via non-carrier bootstrap resampling with replacement (`sample_from="noncarriers"`, `rng.integers`).
2. **Conventional PheWAS:** Performs Firth bias-reduced logistic regression (1-df penalized likelihood-ratio test) adjusted for age, sex, and genetic principal components (`PC1`–`PC10`), retaining two-sided Fisher's exact test only as an exception fallback.
3. **ACAT Aggregation:** Combines aligned p-values from EmbedPheScan and conventional PheWAS using numerically stable equal-weight Aggregated Cauchy Association Test (ACAT) to produce a unified ranking.

## Repository Structure

```text
code/
├── EA_PheWAS.py             # Main execution wrapper for the pipeline
├── EmbedPheScan.py          # Embedding-based p-values generation (non-carrier bootstrap)
├── PheWAS.py                # Covariate-adjusted Firth logistic regression & Fisher fallback
├── ACAT_PheWAS.py           # Embedding/phecode helpers & numerically stable ACAT combination
└── eval_hit_precision.py    # Precision@K evaluation against HPO databases

embedding_model/
└── embed_w2v.py             # Phenotype embedding generation based on Word2Vec model

sample_data/
├── cov_adjust_sample.txt        # Synthetic covariates (eid, age_recruit, sex, PC1..PC10)
├── hesin_sample.txt             # Synthetic diagnosis records (eid, diag_icd10)
├── carrier_sample.txt           # Synthetic gene carrier IDs (sample)
├── icd2phecode_des.csv          # Full public ICD-10 to PheCode 1.2 mapping
├── Phecode_map12_filtered.csv   # Full public PheCode 1.2 description table
└── hpo-phecode1.2_links.tsv     # Full public HPO-to-PheCode 1.2 reference links
```

## Quickstart

Install dependencies (`numpy`, `pandas`, `scipy`, `statsmodels`, `gensim`, `pyhpo`) and run end-to-end on the bundled synthetic `sample_data/` (no external embeddings or UK Biobank files required; `embed_w2v.py` trains ICD-10 embeddings automatically when `--icd_embed` is omitted):

```bash
python code/EA_PheWAS.py --gene PKD1 --out_dir ./results
```

### Helper Variables & Functions

All intermediate dictionary objects used in `code/EA_PheWAS.py` are constructed using helpers in `code/ACAT_PheWAS.py` and `embedding_model/embed_w2v.py`:
- **`loaded_embeddings`** (`{"ind_<embed_name>": {eid: vector}}`): built via `get_group_embed(eid_use_phewas, hesin_diag, icd_embed)` or loaded via `--ind_embed`.
- **`phecode_embeddings`** (`{"icd_<embed_name>": {phecode: vector}}`): built via `make_weighted_phecode_embeddings(icd2phecode, icd_embed, freq_df=...)`.
- **`phecode_mapping`** (`{phecode: "<phecode>: <PhecodeString>"}`): built via `build_phecode_mapping(phecode_df)`.
