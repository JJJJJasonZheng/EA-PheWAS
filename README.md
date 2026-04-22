# EA-PheWAS: Integrating Phenotype Embeddings with PheWAS for Enhanced Gene-Phenotype Discovery

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.04.21.720031-blue.svg)](https://doi.org/10.64898/2026.04.21.720031)

**EA-PheWAS** (Embedding-Augmented PheWAS) is a unified framework that integrates signals from regression-based PheWAS and embedding-similarity–based phenotype prioritization.

## Overview

The pipeline leverages a dual-branch architecture to maximize statistical power and robustness:
1. **EmbedPheScan:** Computes cosine similarities between individual gene carrier embeddings and phenotype embeddings, generating fast permutation-based *z*-score p-values.
2. **Conventional PheWAS:** Performs traditional covariate-adjusted logistic regression, dynamically falling back to Fisher's exact tests for phenotypes with ultra-low carrier counts.
3. **ACAT Aggregation:** Merges the continuous approximation p-values from the embedding space with the categorical p-values from the regression model using the Aggregated Cauchy Association Test (ACAT) to create a single, unified ranking.
4. **Evaluation:** Automatically benchmarks the aggregated results against Human Phenotype Ontology (HPO) ground truth using Precision@K metrics.

## Repository Structure

```text
code/
├── EA_PheWAS.py             # Main execution wrapper for the pipeline
├── EmbedPheScan.py          # Embedding-based p-values generation
├── PheWAS.py                # Covariate-adjusted logistic regression & Fisher exact tests
├── ACAT_PheWAS.py           # Cauchy combination of p-values (ACAT)
└── eval_hit_precision.py    # Precision@K evaluation against HPO databases
 
embedding_model/
└── embed_w2v.py             # Phenotype embedding generation based on Word2Vec model

sample_data/
├── cov_adjust_sample.txt    # Covariates sample data  
└── hesin_sample.txt         # Phenotype list sample data
```