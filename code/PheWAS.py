import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def run_carrierlist_phewas_hybrid_cov(carrier_eids, hesin_use, col, eid_use, cov_adjust, icd_mapping, min_cases, min_car_cases, two_sided):
    cov = cov_adjust[cov_adjust["eid"].isin(set(eid_use))].copy()
    cov["sex_bin"] = cov["sex"].map({"Female": 0, "Male": 1}).fillna(cov["sex"]).astype(float)
    cov = cov.sort_values("eid").set_index("eid")
    
    all_eids = cov.index.to_numpy()
    X_car = np.isin(all_eids, list(set(carrier_eids))).astype(float)
    X_design = sm.add_constant(np.column_stack([X_car, cov["age_recruit"].to_numpy(float), cov["sex_bin"].to_numpy(float)]))
    
    icds = hesin_use[col].dropna().unique()
    rows = []
    
    for icd in icds:
        Y_icd = np.isin(all_eids, hesin_use.loc[hesin_use[col] == icd, "eid"]).astype(float)
        n_case, n_ctrl = int(Y_icd.sum()), int(len(Y_icd) - Y_icd.sum())
        
        if n_case < min_cases: 
            continue
        
        cc = int(((Y_icd == 1) & (X_car == 1)).sum())
        cnc = int(((Y_icd == 1) & (X_car == 0)).sum())
        ctrl_c = int(((Y_icd == 0) & (X_car == 1)).sum())
        ctrl_nc = int(((Y_icd == 0) & (X_car == 0)).sum())
        
        if cc == 0:
            rows.append({col: icd, "beta": np.nan, "se": np.nan, "pval": 1.0, "method": "none", "n_case": n_case, "n_ctrl": n_ctrl, "cc": cc, "cnc": cnc, "ctrl_c": ctrl_c, "ctrl_nc": ctrl_nc})
            continue

        method, beta, se, pval = "fisher", np.nan, np.nan, np.nan
        alt = "two-sided" if two_sided else "greater"
        OR, p_fish = fisher_exact([[cc, ctrl_c], [cnc, ctrl_nc]], alternative=alt)
        beta_fish = np.log(OR) if OR > 0 and np.isfinite(OR) else np.nan
        
        if cc >= min_car_cases:
            try:
                fit = sm.Logit(Y_icd, X_design).fit(disp=False, maxiter=200, method="lbfgs")
                beta, se, pval, method = float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1]), "logit"
            except:
                beta, pval, method = beta_fish, float(p_fish), "fisher_fallback"
        else:
            beta, pval = beta_fish, float(p_fish)

        rows.append({col: icd, "beta": beta, "se": se, "pval": pval, "method": method, "n_case": n_case, "n_ctrl": n_ctrl, "cc": cc, "cnc": cnc, "ctrl_c": ctrl_c, "ctrl_nc": ctrl_nc})

    res = pd.DataFrame(rows)
    if res.empty: 
        return res

    v = res["pval"].notna()
    res["p_bonf"] = np.nan
    res.loc[v, "p_bonf"] = np.clip(res.loc[v, "pval"] * v.sum(), 0, 1)
    
    res["p_fdr"] = np.nan
    if v.sum() > 0:
        res.loc[v, "p_fdr"] = multipletests(res.loc[v, "pval"], method="fdr_bh")[1]

    res["icd_des"] = res[col].map(icd_mapping)
    return res.sort_values("pval").reset_index(drop=True)