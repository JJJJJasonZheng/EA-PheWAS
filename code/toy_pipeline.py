"""Minimal step-by-step toy pipeline showing how to run EA-PheWAS on custom data."""

import os
import sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from EmbedPheScan import phewas_by_embedding_with_zpvalues
from PheWAS import run_carrierlist_phewas_hybrid_cov
from ACAT_PheWAS import (
    add_acat_pvalues_from_embed_and_lr,
    acat_df_to_combined_ranking,
    build_phecode_mapping,
    get_gene_carriers_id,
    get_group_embed,
    make_weighted_phecode_embeddings,
)
from embedding_model.embed_w2v import train_icd_embeddings


def run_toy_custom_pipeline(
    hesin_path=os.path.join(ROOT_DIR, "sample_data/hesin_sample.txt"),
    cov_path=os.path.join(ROOT_DIR, "sample_data/cov_adjust_sample.txt"),
    carrier_path=os.path.join(ROOT_DIR, "sample_data/carrier_sample.txt"),
    icd2phecode_path=os.path.join(ROOT_DIR, "sample_data/icd2phecode_des.csv"),
    phecode_map_path=os.path.join(ROOT_DIR, "sample_data/Phecode_map12_filtered.csv"),
    embed_name="custom_w2v",
    min_cases=5,
    B=1000,
    seed=0,
):
    # 1. Read custom cohort files:
    #    - hesin: tab-separated with columns ['eid', 'diag_icd10']
    #    - cov_adjust: tab-separated with columns ['eid', 'age_recruit', 'sex'] (+ optional 'PC1'..'PC10')
    #    - carrier_file: whitespace-separated with column ['sample'] containing carrier eids
    hesin_diag = pd.read_csv(hesin_path, sep="\t")
    cov_adjust = pd.read_csv(cov_path, sep="\t")

    # 2. Load ICD-10 -> PheCode and PheCode description tables, then map diagnoses to phecodes
    icd2phecode = pd.read_csv(icd2phecode_path)
    phecode_mapping = build_phecode_mapping(pd.read_csv(phecode_map_path))

    icd_to_phecode = (
        icd2phecode.dropna(subset=["diag_icd10", "PheCode"])
        .drop_duplicates("diag_icd10")
        .assign(PheCode=lambda x: x["PheCode"].astype(str).str.strip())
        .set_index("diag_icd10")["PheCode"]
        .to_dict()
    )
    hesin_use = (
        hesin_diag.assign(phecode=hesin_diag["diag_icd10"].map(icd_to_phecode))[["eid", "phecode"]]
        .dropna()
        .drop_duplicates()
    )
    eid_use = hesin_use["eid"].drop_duplicates().tolist()
    carriers = get_gene_carriers_id(carrier_path, hesin_use)

    # 3. Train ICD-10 embeddings from diagnosis records (or load precomputed embeddings)
    #    and build loaded_embeddings and phecode_embeddings
    df_icd_emb = train_icd_embeddings(hesin_diag, size=64, window=5, min_count=1, seed=seed)
    icd_embed = {str(code): row.values for code, row in df_icd_emb.iterrows()}

    icd_counts = hesin_diag["diag_icd10"].astype(str).value_counts().rename_axis("code").to_frame("Freq")
    loaded_embeddings = {f"ind_{embed_name}": get_group_embed(eid_use, hesin_diag, icd_embed)}
    phecode_embeddings = {
        f"icd_{embed_name}": make_weighted_phecode_embeddings(icd2phecode, icd_embed, freq_df=icd_counts)
    }

    # 4. Branch 1: EmbedPheScan (samples pseudo-carriers without replacement from the full cohort)
    embed_res = phewas_by_embedding_with_zpvalues(
        carrier_ids=carriers,
        embed_name=embed_name,
        loaded_embeddings=loaded_embeddings,
        icd_parent_embeddings=phecode_embeddings,
        icd_mapping=phecode_mapping,
        B=B,
        seed=seed,
        sample_from="all",
        replace=False,
    )

    # 5. Branch 2: Conventional PheWAS (Firth logistic regression adjusted for age, sex, PC1..PC10)
    phewas_res = run_carrierlist_phewas_hybrid_cov(
        carrier_eids=carriers,
        hesin_use=hesin_use,
        col="phecode",
        eid_use=eid_use,
        cov_adjust=cov_adjust,
        icd_mapping=phecode_mapping,
        min_cases=min_cases,
    )

    # 6. Combine with ACAT
    acat_df = add_acat_pvalues_from_embed_and_lr(embed_res, phewas_res)
    combined_ranking = acat_df_to_combined_ranking(acat_df, keep_all=True)
    return acat_df, combined_ranking


if __name__ == "__main__":
    acat_df, ranking = run_toy_custom_pipeline()
    cols = [c for c in ["phecode", "icd_des", "p_embed", "p_lr", "p_acat", "rank_combined"] if c in ranking.columns]
    print(ranking[cols].head(10).to_string(index=False))
