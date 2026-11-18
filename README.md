## EA-PheWAS: Integrating Phenotype Embeddings with PheWAS for Enhanced Gene-Phenotype Discovery

[![DOI](https://img.shields.io/badge/DOI-10.64898%2F2026.04.21.720031-blue.svg)](https://doi.org/10.64898/2026.04.21.720031)

**EA-PheWAS** (Embedding-Augmented PheWAS) combines conventional regression-based PheWAS with phenotype-embedding similarity scans for gene-level phenotype prioritization.

## Overview

1. **EmbedPheScan:** Computes cosine similarities between gene carrier embeddings and phecode embeddings, sampling pseudo-carrier groups without replacement from the full cohort (`sample_from="all"`, `replace=False`) to calculate *z*-score p-values.
2. **Conventional PheWAS:** Fits Firth penalized logistic regression adjusted for age, sex, and `PC1`–`PC10`, with Fisher's exact test as a fallback when needed.
3. **ACAT Aggregation:** Combines p-values from EmbedPheScan and conventional PheWAS with the Aggregated Cauchy Association Test (ACAT).

## Repository Structure

```text
code/
├── EA_PheWAS.py             # CLI script for running a single-gene scan + HPO evaluation
├── toy_pipeline.py          # Step-by-step toy pipeline for custom cohort data
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

## Quickstart

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `gensim`, `pyhpo`.

Run on the example data:

```bash
python code/EA_PheWAS.py --gene PKD1 --out_dir ./results
python code/toy_pipeline.py
```

## Running on Custom Data (Toy Pipeline)

Because `sample_data/` includes the full ICD-10, PheCode 1.2, and HPO mapping files, you only need three cohort tables to run EA-PheWAS on your own data:

1. **Diagnosis table (`--hesin`):** tab-separated file with columns `eid` and `diag_icd10` (one row per participant–ICD-10 record).
2. **Covariate table (`--cov`):** tab-separated file with columns `eid`, `age_recruit`, `sex` (`Male`/`Female` or `1`/`0`), and optional `PC1`–`PC10`.
3. **Carrier list (`--carrier_file`):** whitespace-separated file with header `sample` listing rare-variant carrier `eid`s for the target gene.

### Option 1: Command line

```bash
python code/EA_PheWAS.py \
  --gene YOUR_GENE \
  --hesin /path/to/hesin.txt \
  --cov /path/to/cov_adjust.txt \
  --carrier_file /path/to/carriers.txt \
  --min_cases 20 --B 10000 --seed 0 \
  --out_dir ./results
```

(Optional: pass `--icd_embed`, `--ind_embed`, and `--icd_freq` if you already have precomputed ICD-10/individual embeddings.)

### Option 2: Step-by-step Python workflow (`code/toy_pipeline.py`)

`code/toy_pipeline.py` shows how each helper object (`loaded_embeddings`, `phecode_embeddings`, `phecode_mapping`) is built before calling the three steps:

```python
import sys
import pandas as pd

sys.path.insert(0, "code")
from embedding_model.embed_w2v import train_icd_embeddings
from EmbedPheScan import phewas_by_embedding_with_zpvalues
from PheWAS import run_carrierlist_phewas_hybrid_cov
from ACAT_PheWAS import (
    add_acat_pvalues_from_embed_and_lr,
    build_phecode_mapping,
    get_gene_carriers_id,
    get_group_embed,
    make_weighted_phecode_embeddings,
)

# 1. Load custom cohort tables and bundled ontology mappings
hesin_diag = pd.read_csv("sample_data/hesin_sample.txt", sep="\t")
cov_adjust = pd.read_csv("sample_data/cov_adjust_sample.txt", sep="\t")
icd2phecode = pd.read_csv("sample_data/icd2phecode_des.csv")
phecode_mapping = build_phecode_mapping(pd.read_csv("sample_data/Phecode_map12_filtered.csv"))

icd_to_phe = (
    icd2phecode.dropna(subset=["diag_icd10", "PheCode"])
    .drop_duplicates("diag_icd10")
    .assign(PheCode=lambda x: x["PheCode"].astype(str).str.strip())
    .set_index("diag_icd10")["PheCode"].to_dict()
)
hesin_use = hesin_diag.assign(phecode=hesin_diag["diag_icd10"].map(icd_to_phe))[["eid", "phecode"]].dropna().drop_duplicates()
eid_use = hesin_use["eid"].drop_duplicates().tolist()
carriers = get_gene_carriers_id("sample_data/carrier_sample.txt", hesin_use)

# 2. Train ICD-10 embeddings and construct loaded_embeddings & phecode_embeddings
df_icd_emb = train_icd_embeddings(hesin_diag, size=64, seed=0)
icd_embed = {str(k): row.values for k, row in df_icd_emb.iterrows()}
icd_counts = hesin_diag["diag_icd10"].astype(str).value_counts().rename_axis("code").to_frame("Freq")

embed_name = "custom_w2v"
loaded_embeddings = {f"ind_{embed_name}": get_group_embed(eid_use, hesin_diag, icd_embed)}
phecode_embeddings = {f"icd_{embed_name}": make_weighted_phecode_embeddings(icd2phecode, icd_embed, freq_df=icd_counts)}

# 3. Run EmbedPheScan, Firth PheWAS, and ACAT
embed_res = phewas_by_embedding_with_zpvalues(
    carriers, embed_name, loaded_embeddings, phecode_embeddings, phecode_mapping,
    B=1000, seed=0, sample_from="all", replace=False,
)
phewas_res = run_carrierlist_phewas_hybrid_cov(
    carriers, hesin_use, "phecode", eid_use, cov_adjust, phecode_mapping, min_cases=5,
)
acat_df = add_acat_pvalues_from_embed_and_lr(embed_res, phewas_res)
```
