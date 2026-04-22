import pickle
import re
import numpy as np
import pandas as pd

_code_re = re.compile(r"(\d+(?:\.\d+)?)")

def _norm_phecode(x):
    if not x: return None
    m = _code_re.search(str(x).strip())
    return m.group(1) if m else None

def _unique_keep_order(seq):
    seen = set()
    return [x for x in seq if x is not None and not (x in seen or seen.add(x))]

def _precision_at_k(ranked_codes, truth_set, k):
    return sum(c in truth_set for c in ranked_codes[:k]) / k if len(ranked_codes) >= k else np.nan

def evaluate_gene_from_summary_pkl(pkl_path, ground_truth_phecodes, truth_code_col="phecode1.2_code", k_values=(5, 10, 15), strict=False):
    with open(pkl_path, "rb") as f:
        b = pickle.load(f)

    truth_set = set(filter(None, ground_truth_phecodes[truth_code_col].dropna().astype(str).apply(_norm_phecode)))

    raw_c = _unique_keep_order(b["phewas_res"].sort_values("pval_icd")["icd_parent"].apply(_norm_phecode))
    embed_c = _unique_keep_order([_norm_phecode(str(d.get("phenotype", "")).split(":")[0]) for d in sorted(b["embed_results"], key=lambda x: float(x["pval"]))])
    
    agg = b["combined_ranking"]
    if "rank_combined" in agg.columns:
        agg = agg.sort_values("rank_combined")
    agg_c = _unique_keep_order(agg["phecode"].apply(_norm_phecode))

    df = pd.DataFrame([
        {"Method": m, **{f"Top{k}": _precision_at_k(c, truth_set, k) for k in k_values}}
        for m, c in [("Raw PheWAS", raw_c), ("EmbedPheScan", embed_c), ("Aggregated", agg_c)]
    ]).set_index("Method").astype(float).round(3)

    beats = {}
    for k in k_values:
        a, e, p = df.loc["Aggregated", f"Top{k}"], df.loc["EmbedPheScan", f"Top{k}"], df.loc["Raw PheWAS", f"Top{k}"]
        beats[k] = None if np.isnan(a) or np.isnan(e) or np.isnan(p) else ((a > e and a > p) if strict else (a >= e and a >= p))

    valid_beats = [v for v in beats.values() if v is not None]
    return df, {
        "per_k": beats,
        "agg_beats_both_count": sum(valid_beats),
        "n_k": len(k_values),
        "agg_beats_both_all_k": all(valid_beats) if valid_beats else False
    }