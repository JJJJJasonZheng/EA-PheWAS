import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, fisher_exact
from statsmodels.stats.multitest import multipletests

def _penalized_ll(X, y, beta):
    eta = X @ beta
    mu = expit(eta)
    w = np.maximum(mu * (1.0 - mu), np.finfo(float).eps)
    info = X.T @ (w[:, None] * X)
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf, mu, w, info
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5 * logdet), mu, w, info

def _fit_firth(X, y, beta_start=None, fixed_idx=None, max_iter=100, tol=1e-8, max_step=5.0):
    k = X.shape[1]
    beta = np.zeros(k) if beta_start is None else beta_start.copy()
    free = np.arange(k) if fixed_idx is None else np.delete(np.arange(k), fixed_idx)
    if fixed_idx is not None:
        beta[fixed_idx] = 0.0
    old_pll, _, _, _ = _penalized_ll(X, y, beta)
    converged = False
    for _ in range(max_iter):
        _, mu, w, info = _penalized_ll(X, y, beta)
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            info_inv = np.linalg.pinv(info)
        lev = w * np.einsum("ij,jk,ik->i", X, info_inv, X, optimize=True)
        score = X.T @ (y - mu + lev * (0.5 - mu))
        try:
            step = np.linalg.solve(info[np.ix_(free, free)], score[free])
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info[np.ix_(free, free)]) @ score[free]
        mstep = float(np.max(np.abs(step))) if step.size else 0.0
        if mstep > max_step:
            step *= max_step / mstep
        cand = beta.copy()
        cand[free] += step
        new_pll, _, _, _ = _penalized_ll(X, y, cand)
        for _ in range(30):
            if new_pll >= old_pll:
                break
            step *= 0.5
            cand = beta.copy()
            cand[free] += step
            new_pll, _, _, _ = _penalized_ll(X, y, cand)
        diff = new_pll - old_pll
        beta, old_pll = cand, new_pll
        if np.max(np.abs(step)) < tol and abs(diff) < tol:
            converged = True
            break
    _, _, _, info = _penalized_ll(X, y, beta)
    try:
        cov_mat = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov_mat = np.linalg.pinv(info)
    return beta, np.sqrt(np.maximum(np.diag(cov_mat), 0.0)), old_pll, converged

def run_carrierlist_phewas_hybrid_cov(carrier_eids, hesin_use, col="phecode", eid_use=None, cov_adjust=None, icd_mapping=None, min_cases=20, min_car_cases=3, two_sided=True, n_pcs=10):
    del min_car_cases  # retained for backwards compatibility
    cov = cov_adjust[cov_adjust["eid"].isin(set(eid_use))].copy()
    cov["sex_bin"] = cov["sex"].map({"Female": 0.0, "Male": 1.0, "F": 0.0, "M": 1.0}).fillna(pd.to_numeric(cov["sex"], errors="coerce"))
    cov_cols = ["age_recruit", "sex_bin"] + [f"PC{i}" for i in range(1, n_pcs + 1) if f"PC{i}" in cov.columns]
    cov = cov[["eid"] + cov_cols].drop_duplicates("eid").dropna().sort_values("eid").set_index("eid")
    for c in cov_cols:
        vals = cov[c].to_numpy(float)
        if not (np.unique(vals).size <= 2 and set(np.unique(vals)).issubset({0.0, 1.0})):
            cov[c] = (vals - vals.mean()) / vals.std()

    all_eids = cov.index.to_numpy()
    X_car = np.isin(all_eids, list(set(carrier_eids))).astype(float)
    X_design = np.column_stack([np.ones(len(all_eids)), X_car, cov[cov_cols].to_numpy(float)])

    icds = hesin_use[col].dropna().unique()
    rows = []
    alt = "two-sided" if two_sided else "greater"

    for icd in icds:
        Y_icd = np.isin(all_eids, hesin_use.loc[hesin_use[col] == icd, "eid"]).astype(float)
        n_case, n_ctrl = int(Y_icd.sum()), int(len(Y_icd) - Y_icd.sum())
        if n_case < min_cases or n_ctrl == 0:
            continue

        cc = int(((Y_icd == 1) & (X_car == 1)).sum())
        cnc = int(((Y_icd == 1) & (X_car == 0)).sum())
        ctrl_c = int(((Y_icd == 0) & (X_car == 1)).sum())
        ctrl_nc = int(((Y_icd == 0) & (X_car == 0)).sum())

        try:
            if np.linalg.matrix_rank(X_design) < X_design.shape[1]:
                raise ValueError("Rank-deficient design matrix")
            params, bse, pll, ok = _fit_firth(X_design, Y_icd)
            _, _, null_pll, ok0 = _fit_firth(X_design, Y_icd, beta_start=params, fixed_idx=1)
            if not (ok and ok0):
                raise RuntimeError("Firth did not converge")
            beta, se, pval, method = float(params[1]), float(bse[1]), float(chi2.sf(max(0.0, 2.0 * (pll - null_pll)), df=1)), "firth_plrt"
        except Exception:
            OR, p_fish = fisher_exact([[cc, ctrl_c], [cnc, ctrl_nc]], alternative=alt)
            beta = np.log(OR) if OR > 0 and np.isfinite(OR) else np.nan
            se, pval, method = np.nan, float(p_fish), "fisher_fallback"

        rows.append({
            "icd_parent": icd, "beta_icd": beta, "se_icd": se, "pval_icd": pval, "method": method,
            "n_case": n_case, "n_ctrl": n_ctrl, "case_carrier": cc, "case_noncarrier": cnc,
            "ctrl_carrier": ctrl_c, "ctrl_noncarrier": ctrl_nc,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    v = res["pval_icd"].notna()
    res["p_bonf_icd"] = np.nan
    res.loc[v, "p_bonf_icd"] = np.clip(res.loc[v, "pval_icd"] * v.sum(), 0, 1)

    res["p_fdr_icd"] = np.nan
    if v.sum() > 0:
        res.loc[v, "p_fdr_icd"] = multipletests(res.loc[v, "pval_icd"], method="fdr_bh")[1]

    res["icd_des"] = res["icd_parent"].map(icd_mapping) if icd_mapping is not None else res["icd_parent"]
    return res.sort_values("pval_icd").reset_index(drop=True)
