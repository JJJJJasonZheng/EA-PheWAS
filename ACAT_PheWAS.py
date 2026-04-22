import re
import numpy as np
import pandas as pd

def get_ind_embed(eid_ind, icd_embed, hesin):
    icds = hesin[hesin['eid'] == eid_ind]['diag_icd10']
    dim = len(next(iter(icd_embed.values())))
    if len(icds) == 0: 
        return np.zeros(dim)
    embeds = [icd_embed[icd] for icd in icds if icd in icd_embed]
    return np.mean(embeds, axis=0) if embeds else np.zeros(dim)

def get_disease_embed_fromicd(icd_disease, icd_embed, hesin):
    eids = hesin[hesin['diag_icd10'].isin(icd_disease)]['eid'].unique()
    return {eid: get_ind_embed(eid, icd_embed, hesin) for eid in eids}

def get_disease_embed_fromind(icd_disease, ind_embed, hesin):
    eids = hesin[hesin['diag_icd10'].isin(icd_disease)]['eid'].unique()
    return {eid: ind_embed[eid] for eid in eids if eid in ind_embed}

def get_gene_carriers_id(carrier_file, hesin):
    c_ids = pd.read_csv(carrier_file, sep=' ')['sample'].tolist()
    return list(set(c_ids) & set(hesin['eid']))

def get_gene_carriers_embed(carrier_file, icd_embed, hesin):
    c_ids = get_gene_carriers_id(carrier_file, hesin)
    if not c_ids: 
        return np.zeros(len(next(iter(icd_embed.values()))))
    return np.mean([get_ind_embed(e, icd_embed, hesin) for e in c_ids], axis=0)

def get_gene_carriers_embed_fromind(carrier_file, ind_embed, hesin):
    c_ids = get_gene_carriers_id(carrier_file, hesin)
    return {e: ind_embed[e] for e in c_ids if e in ind_embed}

def get_group_embed(eid_group, hesin_group, embed_use):
    return {e: get_ind_embed(e, embed_use, hesin_group) for e in eid_group}
    
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def _acat_two(p_embed, p_lr, w_embed=0.5, w_lr=0.5, p_min=1e-300, p_max=1-1e-16):
    p1 = np.clip(np.asarray(p_embed, dtype=float), p_min, p_max)
    p2 = np.clip(np.asarray(p_lr, dtype=float), p_min, p_max)
    w = np.array([w_embed, w_lr], dtype=float)
    w /= np.sum(w)
    T = w[0] * np.tan(np.pi * (0.5 - p1)) + w[1] * np.tan(np.pi * (0.5 - p2))
    return np.clip(0.5 - np.arctan(T) / np.pi, 0.0, 1.0)

_code_re = re.compile(r"(\d+(?:\.\d+)?)")

def _extract_phecode(x):
    if x is None: return None
    m = _code_re.search(str(x).strip())
    return m.group(1) if m else None

def add_acat_pvalues_from_embed_and_lr(embed_results, phewas_res, w_embed=0.5, w_lr=0.5, embed_p_col="pval", embed_label_col="phenotype", lr_p_col="pval_icd", lr_code_col="icd_parent", keep_cols_lr=None, keep_cols_embed=None, sort_by="p_acat"):
    emb = pd.DataFrame(embed_results).dropna(subset=[embed_label_col, embed_p_col]).copy()
    emb["phecode"] = emb[embed_label_col].apply(_extract_phecode)
    emb["p_embed"] = pd.to_numeric(emb[embed_p_col], errors="coerce")
    emb = emb.dropna(subset=["phecode", "p_embed"])
    emb = emb[["phecode", "p_embed"] + (keep_cols_embed or [c for c in ["similarity", "centered_similarity", "z", "qval"] if c in emb.columns])]

    lr = phewas_res.dropna(subset=[lr_code_col, lr_p_col]).copy()
    lr["phecode"] = lr[lr_code_col].apply(_extract_phecode)
    lr["p_lr"] = pd.to_numeric(lr[lr_p_col], errors="coerce")
    lr = lr.dropna(subset=["phecode", "p_lr"])
    lr = lr[["phecode", "p_lr"] + (keep_cols_lr or [c for c in ["icd_des", "beta_icd", "se_icd", "n_case", "case_carrier"] if c in lr.columns])]

    merged = lr.merge(emb, on="phecode", how="inner")
    merged["p_acat"] = _acat_two(merged["p_embed"], merged["p_lr"], w_embed, w_lr)
    
    if sort_by in merged.columns:
        merged = merged.sort_values(sort_by).reset_index(drop=True)
    return merged

def acat_df_to_combined_ranking(acat_df, phecode_col="phecode", p_acat_col="p_acat", rank_embed_col="rank_embed", rank_lr_col="rank_lr", keep_all=True):
    df = acat_df.dropna(subset=[phecode_col, p_acat_col]).copy()
    df["phecode"] = df[phecode_col].apply(_extract_phecode)
    df = df.dropna(subset=["phecode"])
    
    df["rank_embed"] = pd.to_numeric(df.get(rank_embed_col, df.get("p_embed")), errors="coerce").rank(method="average")
    df["rank_lr"] = pd.to_numeric(df.get(rank_lr_col, df.get("p_lr")), errors="coerce").rank(method="average")
    df["rank_combined"] = pd.to_numeric(df[p_acat_col], errors="coerce").rank(method="average")
    
    df = df.dropna(subset=["rank_embed", "rank_lr", "rank_combined"]).sort_values("rank_combined").reset_index(drop=True)
    
    cols_first = ["phecode", "rank_embed", "rank_lr", "rank_combined"]
    return df[cols_first + [c for c in df.columns if c not in cols_first]] if keep_all else df[cols_first]