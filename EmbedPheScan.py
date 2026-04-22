import numpy as np
from scipy.stats import norm

def _cosine_sim_vector(mean_vec, mat, mat_norm):
    v = mean_vec.astype(float)
    v_norm = np.linalg.norm(v)
    if v_norm <= 0: return np.full(mat.shape[0], np.nan)
    return (mat @ v) / (mat_norm * v_norm)

def _norm_sf(z):
    return norm.sf(np.asarray(z, dtype=float))

def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    q = (p[order] * m) / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out

def phewas_by_embedding_with_zpvalues(carrier_ids, embed_name, loaded_embeddings, icd_parent_embeddings, icd_mapping, B=10000, seed=0, sample_from="noncarriers", center_profile=True, center_func="mean", two_sided=False, min_sd=1e-8):
    ind_embed = loaded_embeddings[f"ind_{embed_name}"]
    icd_embed = icd_parent_embeddings[f"icd_{embed_name}"]
    
    carrier_vecs = np.array([ind_embed[eid] for eid in carrier_ids if eid in ind_embed and ind_embed[eid] is not None])
    if len(carrier_vecs) == 0: raise ValueError("No valid carriers")
    
    t_obs = carrier_vecs.mean(axis=0)
    n_carriers = len(carrier_vecs)

    all_eids = list(ind_embed.keys())
    all_vecs = np.array([v for v in ind_embed.values() if v is not None])
    
    pool = all_vecs[~np.isin(all_eids, carrier_ids)] if sample_from == "noncarriers" else all_vecs

    phen_names = [icd_mapping.get(k, k) for k, v in icd_embed.items() if v is not None]
    phen_mat = np.array([v for v in icd_embed.values() if v is not None])
    phen_norm = np.linalg.norm(phen_mat, axis=1)
    keep = phen_norm > 0
    phen_mat, phen_norm = phen_mat[keep], phen_norm[keep]
    phen_names = [phen_names[i] for i in np.where(keep)[0]]

    center = np.nanmean if center_func == "mean" else np.nanmedian
    s_obs = _cosine_sim_vector(t_obs, phen_mat, phen_norm)
    s_obs_c = s_obs - center(s_obs) if center_profile else s_obs

    rng = np.random.default_rng(seed)
    null_sum = np.zeros(len(phen_mat))
    null_sumsq = np.zeros(len(phen_mat))

    for _ in range(B):
        idx = rng.integers(0, len(pool), size=n_carriers)
        s_null = _cosine_sim_vector(pool[idx].mean(axis=0), phen_mat, phen_norm)
        if center_profile: s_null -= center(s_null)
        null_sum += s_null
        null_sumsq += s_null * s_null

    null_mean = null_sum / B
    null_sd = np.maximum(np.sqrt(np.maximum((null_sumsq / B) - null_mean**2, 0)), min_sd)

    z = (s_obs_c - null_mean) / null_sd
    pvals = np.minimum(2.0 * _norm_sf(np.abs(z)), 1.0) if two_sided else _norm_sf(z)
    qvals = _bh_fdr(pvals)

    res = [{"phenotype": n, "similarity": float(s), "centered_similarity": float(sc), "z": float(z_), "pval": float(p), "qval": float(q)} 
           for n, s, sc, z_, p, q in zip(phen_names, s_obs, s_obs_c, z, pvals, qvals)]
           
    return sorted(res, key=lambda x: x["pval"])