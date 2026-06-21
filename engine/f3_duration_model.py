"""
F3 - Duration Refinement Model, inside the Calibration Harness
Flipkart GRiD Round 2 | Event-Impact & Resource Recommendation Engine

A gradient-boosted quantile regressor on log-duration (P10/P50/P90 in one pass),
trained on a held-out split and immediately validated for CALIBRATION COVERAGE:
does the P90 actually cover ~90% of unseen events?  Subordinate to F2 - we report
whether it earns its place, honestly, with baselines.

Reads canonical Step 0 data + the F2 reference table.  Saves the model bundle.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
DUR = os.path.join(_BASE, "duration_clean.csv")
REF = os.path.join(_BASE, "reference_table.csv")
BUNDLE = os.path.join(_BASE, "f3_duration_model.pkl")

CAT = ["event_cause", "priority", "event_type", "veh_type", "police_station", "zone"]
NUM = ["on_corridor", "desc_highway", "desc_multiple", "desc_signal", "latitude", "longitude"]
QUANTILES = [0.10, 0.50, 0.90]
SEED = 42


def pinball(y, q_pred, alpha):
    d = y - q_pred
    return np.mean(np.maximum(alpha * d, (alpha - 1) * d))


def medae(y, p):
    return float(np.median(np.abs(np.asarray(y) - np.asarray(p))))


def main():
    df = pd.read_csv(DUR, low_memory=False)
    ref = pd.read_csv(REF)
    ref_median = dict(zip(ref["event_cause"], ref["median_min"]))

    # ---- Features (no leakage: hour-of-day, junction, status, closure, near_24h_cap all excluded)
    for c in NUM:
        df[c] = df[c].astype(float)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xcat = enc.fit_transform(df[CAT].astype(str))        # vocab from full data (not the target)
    X = np.concatenate([Xcat, df[NUM].values], axis=1)
    cat_mask = [True] * len(CAT) + [False] * len(NUM)
    y_log = np.log1p(df["duration_min"].values)
    y = df["duration_min"].values
    causes = df["event_cause"].values

    Xtr, Xte, ytr_log, yte_log, ytr, yte, ctr, cte = train_test_split(
        X, y_log, y, causes, test_size=0.25, random_state=SEED)
    print("=" * 78)
    print(" F3 - QUANTILE DURATION MODEL  +  CALIBRATION HARNESS")
    print("=" * 78)
    print(f"  train={len(Xtr)}  test={len(Xte)}   features: {len(CAT)} categorical + {len(NUM)} numeric")
    print(f"  target: log1p(duration_min)   quantiles: {QUANTILES}")

    # ---- Fit one model per quantile
    models = {}
    for a in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=a, categorical_features=cat_mask,
            learning_rate=0.06, max_iter=400, max_depth=4,
            min_samples_leaf=25, l2_regularization=1.0, random_state=SEED)
        m.fit(Xtr, ytr_log)
        models[a] = m

    # ---- Predict (back to minutes), enforce monotone quantiles
    preds = {a: np.expm1(models[a].predict(Xte)) for a in QUANTILES}
    P = np.vstack([preds[0.10], preds[0.50], preds[0.90]]).T
    n_cross = int((np.diff(P, axis=1) < 0).any(axis=1).sum())
    P = np.sort(P, axis=1)                                # fix any crossing
    p10, p50, p90 = P[:, 0], P[:, 1], P[:, 2]

    # ---- Baselines
    const_pred = np.full(len(yte), np.expm1(np.median(ytr_log)))
    pooled_med = ref["median_min"].median()
    type_pred = np.array([ref_median.get(c, pooled_med) for c in cte])

    print("-" * 78)
    print(" 1) ACCURACY vs BASELINES  (MedAE, minutes - lower is better)")
    print("-" * 78)
    short = yte <= 60                                     # the dense, operationally-common region
    longt = ~short
    def row(name, pred):
        print(f"    {name:<22} overall={medae(yte,pred):6.1f}   "
              f"short(<=60m,n={short.sum()})={medae(yte[short],pred[short]):6.1f}   "
              f"long(>60m,n={longt.sum()})={medae(yte[longt],pred[longt]):6.1f}")
    row("constant (global med)", const_pred)
    row("type-lookup (F2)", type_pred)
    row("model P50", p50)

    print("-" * 78)
    print(" 2) CALIBRATION COVERAGE  (empirical vs nominal - the load-bearing test)")
    print("-" * 78)
    print(f"    {'quantile':<12}{'nominal':>9}{'empirical':>11}{'gap':>8}")
    for a, q in zip(QUANTILES, [p10, p50, p90]):
        cov = float(np.mean(yte <= q))
        print(f"    {('P'+str(int(a*100))):<12}{a:>9.2f}{cov:>11.3f}{cov-a:>8.3f}")
    print(f"    quantile crossings before sort: {n_cross}/{len(yte)} (corrected)")
    print()
    print(f"    {'P90 coverage by region':<28}{'nominal':>9}{'empirical':>11}")
    print(f"    {'  short events (<=60m)':<28}{0.90:>9.2f}{float(np.mean(yte[short]<=p90[short])):>11.3f}")
    print(f"    {'  long events (>60m)':<28}{0.90:>9.2f}{float(np.mean(yte[longt]<=p90[longt])):>11.3f}")

    print("-" * 78)
    print(" 3) PINBALL LOSS  (minutes, lower is better)")
    print("-" * 78)
    for a, q in zip(QUANTILES, [p10, p50, p90]):
        print(f"    P{int(a*100):<3} pinball = {pinball(yte, q, a):6.2f}")

    # ---- Verdict line
    overall_model = medae(yte, p50); overall_type = medae(yte, type_pred)
    short_model = medae(yte[short], p50[short]); short_type = medae(yte[short], type_pred[short])
    print("=" * 78)
    print(" READOUT")
    print("=" * 78)
    print(f"  - Overall MedAE: model {overall_model:.1f} vs F2 lookup {overall_type:.1f} "
          f"-> {'model' if overall_model<overall_type else 'F2'} ahead "
          f"(by {abs(overall_model-overall_type):.1f} min)")
    print(f"  - Short events : model {short_model:.1f} vs F2 lookup {short_type:.1f} "
          f"-> {'model wins' if short_model<short_type else 'F2 wins'} where most events live")
    print(f"  - P90 coverage : {float(np.mean(yte<=p90)):.1%} (target 90%) "
          f"-> {'calibrated' if abs(np.mean(yte<=p90)-0.9)<0.04 else 'NEEDS WORK'}")

    with open(BUNDLE, "wb") as f:
        pickle.dump({"models": models, "encoder": enc, "cat": CAT, "num": NUM,
                     "cat_mask": cat_mask, "quantiles": QUANTILES}, f)
    print(f"\n  saved model bundle -> {BUNDLE}")


if __name__ == "__main__":
    main()
