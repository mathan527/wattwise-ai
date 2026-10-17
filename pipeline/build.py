"""WattWise AI - offline ML pipeline.

Reads the UCI Electricity Load Diagrams 2011-2014 file, runs the Unit 3 unsupervised
methods (K-means, GMM+EM, PCA, Factor Analysis, hierarchical clustering, evaluation)
and the multi-signal anomaly scoring, and writes web/data/results.json for the dashboard.

Usage:  python pipeline/build.py [path/to/LD2011_2014.txt]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data_raw" / "uci" / "LD2011_2014.txt"
OUT = ROOT / "web" / "data" / "results.json"
RNG = 42
N_METERS = 20
TRAIN_END, VAL_END = "2014-06-30 23:00", "2014-09-30 23:00"  # train Jan-Jun, validation Jul-Sep, test Oct-Dec
WEIGHTS = {"baseline": 0.5, "centroid": 0.3, "gmm": 0.2}
VAL_QUANTILE = 0.995


def r(x, n=4):
    return float(np.round(x, n))


def load_hourly():
    df = pd.read_csv(SRC, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.loc["2014-01-01":"2014-12-31"]
    zero_frac = (df == 0).mean()
    ok = zero_frac[(zero_frac < 0.005)].index
    # keep meters with sensible, comparable load (drop the very smallest and largest)
    mean = df[ok].mean()
    ok = mean[(mean > mean.quantile(0.2))].sort_values().index
    pick = ok[np.linspace(0, len(ok) - 1, N_METERS).astype(int)]
    kwh = (df[pick] * 0.25).resample("h").sum()  # kW per 15 min -> kWh per hour
    kwh.columns = [f"M{i + 1:02d}" for i in range(N_METERS)]
    return kwh, list(pick)


def day_matrix(kwh, scale):
    """One row per meter-day: 24 hourly values relative to the meter's training mean."""
    rows, meta = [], []
    for m in kwh.columns:
        s = kwh[m] / scale[m]
        d = s.groupby(s.index.date)
        for day, g in d:
            if len(g) == 24:
                rows.append(g.values)
                meta.append((m, pd.Timestamp(day)))
    X = np.array(rows)
    M = pd.DataFrame(meta, columns=["meter", "day"])
    return X, M


def main():
    kwh, orig_ids = load_hourly()
    train = kwh.loc[:TRAIN_END]
    scale = train.mean()
    X, M = day_matrix(kwh, scale)
    is_train = (M["day"] <= pd.Timestamp(TRAIN_END).normalize()).values
    Xtr = X[is_train]

    # ------------------------------------------------------------ K-means
    ks = list(range(2, 11))
    rng = np.random.RandomState(RNG)
    sub = rng.choice(len(Xtr), min(3000, len(Xtr)), replace=False)
    elbow = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=RNG).fit(Xtr)
        lab = km.labels_
        elbow.append({"k": k, "inertia": r(km.inertia_, 2),
                      "silhouette": r(silhouette_score(Xtr[sub], lab[sub])),
                      "davies_bouldin": r(davies_bouldin_score(Xtr, lab))})
    best_k = max(elbow, key=lambda e: e["silhouette"])["k"]
    K = best_k
    km = KMeans(K, n_init=20, random_state=RNG).fit(Xtr)
    # order clusters by mean level so labels are stable/readable
    order = np.argsort(km.cluster_centers_.mean(1))
    remap = {int(o): i for i, o in enumerate(order)}
    cent = km.cluster_centers_[order]
    km_all = np.array([remap[l] for l in km.predict(X)])
    km_tr = km_all[is_train]

    # ------------------------------------------------------------ GMM + EM
    gmm_sel = []
    for k in range(1, 13):
        g = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=RNG, reg_covar=1e-4).fit(Xtr)
        gmm_sel.append({"k": k, "aic": r(g.aic(Xtr), 1), "bic": r(g.bic(Xtr), 1)})
    gk = min(gmm_sel, key=lambda e: e["bic"])["k"]
    gk_use = max(gk, 2)
    # EM log-likelihood curve: run EM one iteration at a time
    g_it = GaussianMixture(gk_use, covariance_type="diag", max_iter=1, warm_start=True, tol=0,
                           init_params="kmeans", random_state=RNG, reg_covar=1e-4)
    ll_curve = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for it in range(1, 41):
            g_it.fit(Xtr)
            ll_curve.append({"iter": it, "loglik": r(g_it.score(Xtr) * len(Xtr), 1)})
    gmm = GaussianMixture(gk_use, covariance_type="diag", n_init=5, random_state=RNG, reg_covar=1e-4).fit(Xtr)
    post = gmm.predict_proba(X)
    gmm_lab = gmm.predict(X)
    gmm_ll = gmm.score_samples(X)  # per-day log-likelihood

    # ------------------------------------------------------------ PCA
    pca_full = PCA(n_components=10, random_state=RNG).fit(Xtr)
    pca = PCA(n_components=3, random_state=RNG).fit(Xtr)
    Z = pca.transform(X)
    samp = rng.choice(len(X), 1800, replace=False)

    # ------------------------------------------------------------ Factor analysis
    fa = FactorAnalysis(n_components=3, rotation="varimax", random_state=RNG).fit(Xtr)
    F = fa.transform(X)
    loadings = fa.components_.T  # 24 x 3

    # ------------------------------------------------------------ Hierarchical (meter level)
    meter_prof = {}
    for m in kwh.columns:
        s = (kwh[m].loc[:TRAIN_END] / scale[m])
        wk = s[s.index.dayofweek < 5].groupby(s[s.index.dayofweek < 5].index.hour).mean().values
        we = s[s.index.dayofweek >= 5].groupby(s[s.index.dayofweek >= 5].index.hour).mean().values
        meter_prof[m] = (wk, we)
    names = list(kwh.columns)
    P = np.array([np.concatenate(meter_prof[m]) for m in names])
    hier = {}
    for method in ("ward", "average", "complete"):
        Lk = linkage(P, method=method, metric="euclidean")
        hier[method] = {"coph": r(cophenet(Lk, pdist(P))[0], 3)}
    Lw = linkage(P, method="ward")
    dn = dendrogram(Lw, no_plot=True, labels=names)
    cuts = {str(k): fcluster(Lw, k, criterion="maxclust").tolist() for k in range(2, 7)}

    # ------------------------------------------------------------ model comparison + stability
    lab_h = AgglomerativeClustering(K, linkage="ward").fit_predict(Xtr[sub])
    comp = []
    for name, lab, Xe in (("K-means", km_tr[sub], Xtr[sub]),
                          ("GMM (EM)", gmm.predict(Xtr)[sub], Xtr[sub]),
                          ("Hierarchical (Ward)", lab_h, Xtr[sub])):
        comp.append({"model": name, "k": int(len(set(lab))),
                     "silhouette": r(silhouette_score(Xe, lab)),
                     "davies_bouldin": r(davies_bouldin_score(Xe, lab)),
                     "calinski_harabasz": r(calinski_harabasz_score(Xe, lab), 1)})
    ref = KMeans(K, n_init=10, random_state=0).fit_predict(Xtr)
    seed_ari = [r(adjusted_rand_score(ref, KMeans(K, n_init=10, random_state=s).fit_predict(Xtr)), 3) for s in range(1, 6)]
    # temporal stability: fit on Jan-Mar vs Apr-Jun, then compare assignments on Apr-Jun days
    q1 = (M["day"] <= "2014-03-31").values & is_train
    q2 = (M["day"] > "2014-03-31").values & is_train
    ka = KMeans(K, n_init=10, random_state=RNG).fit(X[q1])
    kb = KMeans(K, n_init=10, random_state=RNG).fit(X[q2])
    temporal_ari = r(adjusted_rand_score(ka.predict(X[q2]), kb.predict(X[q2])), 3)
    gm_ari = r(adjusted_rand_score(km_tr, gmm.predict(Xtr)), 3)
    best = max(comp, key=lambda c: c["silhouette"] - c["davies_bouldin"] * 0.25)
    reasoning = (f"{best['model']} gives the best balance of silhouette ({best['silhouette']}) and Davies-Bouldin "
                 f"({best['davies_bouldin']}). K-means is stable across seeds (mean ARI {np.mean(seed_ari):.2f}) and "
                 f"across time periods (ARI {temporal_ari}). GMM is chosen as the probabilistic layer because BIC "
                 f"prefers {gk} component(s) and it supplies membership probabilities used in the anomaly score. "
                 f"K-means vs GMM agreement ARI = {gm_ari}.")

    # ------------------------------------------------------------ anomaly scoring (hour level)
    idx = kwh.index
    wk = (idx.dayofweek >= 5).astype(int)
    long = kwh.stack().rename("kwh").reset_index()
    long.columns = ["ts", "meter", "kwh"]
    long["hour"] = long["ts"].dt.hour
    long["wkend"] = (long["ts"].dt.dayofweek >= 5).astype(int)
    long["day"] = long["ts"].dt.normalize()
    long["split"] = np.where(long["ts"] <= TRAIN_END, "train", np.where(long["ts"] <= VAL_END, "val", "test"))
    trn = long[long["split"] == "train"]
    grp = trn.groupby(["meter", "hour", "wkend"])["kwh"]
    base = grp.median().rename("base")
    mad = grp.apply(lambda s: np.median(np.abs(s - s.median()))).rename("mad")
    st = pd.concat([base, mad], axis=1).reset_index()
    long = long.merge(st, on=["meter", "hour", "wkend"], how="left")
    long["scale"] = long["meter"].map(scale)
    long["sigma"] = np.maximum(1.4826 * long["mad"], 0.03 * long["scale"])
    long["z"] = (long["kwh"] - long["base"]) / long["sigma"]

    # centroid component: hour-level deviation of the day's profile from the day's cluster centroid
    day_key = {(m, d): i for i, (m, d) in enumerate(zip(M["meter"], M["day"]))}
    di = np.array([day_key.get((m, d), -1) for m, d in zip(long["meter"], long["day"])])
    valid = di >= 0
    long = long[valid].reset_index(drop=True)
    di = di[valid]
    cl = km_all[di]
    hr = long["hour"].values
    v = long["kwh"].values / long["scale"].values
    resid = v - cent[cl, hr]
    # per-cluster, per-hour spread measured on training days
    trd = long["split"].values == "train"
    sd = np.ones((K, 24))
    for c in range(K):
        for h in range(24):
            msk = trd & (cl == c) & (hr == h)
            if msk.sum() > 5:
                sd[c, h] = max(resid[msk].std(), 0.03)
    long["cluster"] = cl
    long["cent_z"] = resid / sd[cl, hr]
    long["gmm_ll"] = gmm_ll[di]
    long["gmm_pmax"] = post[di].max(1)
    # GMM component: deviation from the posterior-weighted mixture mean at this hour. Low membership
    # confidence mixes components and widens the expected spread automatically.
    P_ = post[di]
    mu_h = np.einsum("nk,nk->n", P_, gmm.means_[:, hr].T)
    var_h = np.einsum("nk,nk->n", P_, gmm.covariances_[:, hr].T + (gmm.means_[:, hr].T - mu_h[:, None]) ** 2)
    long["gmm_z"] = (v - mu_h) / np.sqrt(var_h + 1e-6)

    # each signal -> 0..1 "severity" = |z| scaled by its 99th percentile on training data (capped at 3x)
    raw = {"baseline": np.abs(long["z"].values), "centroid": np.abs(long["cent_z"].values),
           "gmm": np.abs(long["gmm_z"].values)}
    q99 = {k: float(np.quantile(v_[trd], 0.99)) for k, v_ in raw.items()}

    def squash(k, x):
        return np.clip(x / q99[k], 0, 3) / 3

    ranks = {k: squash(k, v_) for k, v_ in raw.items()}
    score = sum(WEIGHTS[k] * ranks[k] for k in WEIGHTS)
    long["score"] = score
    for k in ranks:
        long[f"r_{k}"] = ranks[k]
    long["excess"] = np.maximum(long["kwh"] - long["base"], 0)

    val = long["split"] == "val"
    thr = float(np.quantile(long.loc[val, "score"], VAL_QUANTILE))
    test = long["split"] == "test"
    flagged = long[test & (long["score"] >= thr)]

    # synthetic-injection check (clearly synthetic: x2.5 spikes on random test hours)
    rs = np.random.RandomState(7)
    ti = np.where(test.values)[0]
    inj = rs.choice(ti, 400, replace=False)
    kk = long["kwh"].values.copy()
    kk[inj] *= 2.5
    z2 = (kk - long["base"].values) / long["sigma"].values
    v2 = kk / long["scale"].values
    cz2 = (v2 - cent[cl, hr]) / sd[cl, hr]
    mu2 = np.einsum("nk,nk->n", P_, gmm.means_[:, hr].T)
    gz2 = (v2 - mu2) / np.sqrt(var_h + 1e-6)
    s2 = (WEIGHTS["baseline"] * squash("baseline", np.abs(z2)) +
          WEIGHTS["centroid"] * squash("centroid", np.abs(cz2)) +
          WEIGHTS["gmm"] * squash("gmm", np.abs(gz2)))
    hit = (s2[inj] >= thr)
    base_flags = (long["score"].values[ti] >= thr).sum()
    injection = {"injected": 400, "multiplier": 2.5, "recall": r(hit.mean(), 3),
                 "false_alarm_rate_clean_test": r(base_flags / len(ti), 4)}
    # single-signal comparison on same injected set (recall at same flag-rate on validation)
    singles = {}
    for name, arr_clean, arr_inj in (("Baseline z only", raw["baseline"], np.abs(z2)),
                                     ("Centroid distance only", raw["centroid"], np.abs(cz2)),
                                     ("GMM posterior only", raw["gmm"], np.abs(gz2))):
        t = np.quantile(arr_clean[val.values], VAL_QUANTILE)
        singles[name] = r((arr_inj[inj] >= t).mean(), 3)
    singles["Combined score"] = injection["recall"]

    cand = long[test].sort_values("score", ascending=False).head(3000)
    cand_out = [{
        "ts": t.strftime("%Y-%m-%d %H:%M"), "m": m, "kwh": r(k, 3), "base": r(b, 3), "z": r(z_, 2),
        "cz": r(c_, 2), "cl": int(c), "pm": r(pm, 3), "rb": r(a, 4), "rc": r(b2, 4), "rg": r(c2, 4),
        "ex": r(e, 3)
    } for t, m, k, b, z_, c_, c, pm, a, b2, c2, e in zip(
        cand["ts"], cand["meter"], cand["kwh"], cand["base"], cand["z"], cand["cent_z"], cand["cluster"],
        cand["gmm_pmax"], cand["r_baseline"], cand["r_centroid"], cand["r_gmm"], cand["excess"])]

    # validation score sample so the UI can recompute thresholds for any weights
    val_df = long[val]
    # random thinning (same rows for all signals); a fixed stride would only hit a subset of meters
    keep = np.sort(np.random.RandomState(3).choice(len(val_df), 12000, replace=False))
    val_scores = {k: np.round(val_df[f"r_{k}"].values[keep], 4).tolist() for k in WEIGHTS}

    # ------------------------------------------------------------ export
    daily = kwh.resample("D").sum()
    cluster_sizes = np.bincount(km_all, minlength=K)
    out = {
        "meta": {
            "dataset": "UCI Electricity Load Diagrams 2011-2014 (ID 321)", "year": 2014,
            "meters": N_METERS, "hours": int(len(kwh)), "meter_days": int(len(X)),
            "split": {"train": "Jan-Jun", "validation": "Jul-Sep", "test": "Oct-Dec"},
            "units": "kWh per hour (15-min kW readings x 0.25, summed)",
            "weights": WEIGHTS, "val_quantile": VAL_QUANTILE, "threshold": r(thr, 4),
        },
        "meters": [{"id": m, "mean_kwh": r(scale[m], 2), "uci_id": str(o)} for m, o in zip(names, orig_ids)],
        "daily": {"dates": [d.strftime("%Y-%m-%d") for d in daily.index],
                  "series": {m: np.round(daily[m].values, 1).tolist() for m in names}},
        "kmeans": {"k": K, "elbow": elbow, "centroids": np.round(cent, 3).tolist(), "sizes": cluster_sizes.tolist(),
                   "weekend_share": [r(M["day"][km_all == c].dt.dayofweek.ge(5).mean(), 3) for c in range(K)]},
        "gmm": {"k": gk_use, "selection": gmm_sel, "ll_curve": ll_curve,
                "weights": np.round(gmm.weights_, 3).tolist(),
                "mean_max_posterior": r(post.max(1).mean(), 3),
                "uncertain_share": r((post.max(1) < 0.8).mean(), 3)},
        "pca": {"explained": np.round(pca_full.explained_variance_ratio_, 4).tolist(),
                "cum": np.round(np.cumsum(pca_full.explained_variance_ratio_), 4).tolist(),
                "scatter": [[r(Z[i, 0], 3), r(Z[i, 1], 3), int(km_all[i]), int(i)] for i in samp],
                "loadings": np.round(pca.components_[:2], 3).tolist()},
        "factor": {"loadings": np.round(loadings, 3).tolist(),
                   "scores": [[r(F[i, 0], 3), r(F[i, 1], 3), int(km_all[i])] for i in samp],
                   "noise_var": np.round(fa.noise_variance_, 3).tolist()},
        "hier": {"icoord": dn["icoord"], "dcoord": dn["dcoord"], "leaves": dn["ivl"], "cuts": cuts,
                 "names": names, "cophenetic": hier},
        "evaluation": {"models": comp, "seed_ari": seed_ari, "temporal_ari": temporal_ari,
                       "kmeans_vs_gmm_ari": gm_ari, "reasoning": reasoning},
        "anomaly": {"threshold": r(thr, 4), "n_flagged": int(len(flagged)),
                    "n_test_hours": int(test.sum()), "injection": injection, "single_signal_recall": singles,
                    "candidates": cand_out, "val_scores": val_scores},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB); K={K} GMM k={gk_use}; flagged {len(flagged)} test hours")
    print(json.dumps({"elbow": elbow, "eval": comp, "injection": injection, "singles": singles}, indent=1))


if __name__ == "__main__":
    main()
