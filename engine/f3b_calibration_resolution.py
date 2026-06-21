"""
F3b - Calibration Resolution
The harness showed the GBM P90 is miscalibrated (79% overall, 52% on long events).
Hypothesis: F2's EMPIRICAL per-type P90 should be better calibrated, because it's
the actual 90th percentile per type rather than a regularized parametric estimate.

Same train/test split as the harness (seed=42, 25% test). We compute the per-type
P90 on TRAIN ONLY and measure coverage on the held-out TEST set, head-to-head with
the model's P90.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
DUR = os.path.join(_BASE, "duration_clean.csv")
BUNDLE = os.path.join(_BASE, "f3_duration_model.pkl")
SEED = 42


def decon_p90(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    kept = s[s <= q3 + 1.5 * (q3 - q1)]
    return (kept if len(kept) else s).quantile(0.90)


def main():
    df = pd.read_csv(DUR, low_memory=False)
    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.25, random_state=SEED)
    tr, te = df.iloc[itr], df.iloc[ite]
    yte = te["duration_min"].values
    short = yte <= 60
    longt = ~short

    # ---- F2 empirical per-type P90, computed on TRAIN ONLY ----
    train_p90 = tr.groupby("event_cause")["duration_min"].apply(decon_p90).to_dict()
    train_n = tr.groupby("event_cause")["duration_min"].size().to_dict()
    pooled_p90 = decon_p90(tr["duration_min"])
    # sparse types (train n < 10) fall back to pooled - mirrors F2
    f2_p90 = np.array([train_p90.get(c, pooled_p90) if train_n.get(c, 0) >= 10 else pooled_p90
                       for c in te["event_cause"]])

    # ---- Model P90 on the same test rows ----
    with open(BUNDLE, "rb") as f:
        b = pickle.load(f)
    Xcat = b["encoder"].transform(te[b["cat"]].astype(str))
    X = np.concatenate([Xcat, te[b["num"]].astype(float).values], axis=1)
    Pm = np.vstack([np.expm1(b["models"][a].predict(X)) for a in b["quantiles"]]).T
    Pm = np.sort(Pm, axis=1)
    model_p90 = Pm[:, 2]

    print("=" * 74)
    print(" F3b - CALIBRATION RESOLUTION:  model P90  vs  F2 empirical P90")
    print("=" * 74)
    print(f"  test events: {len(te)}  (short<=60m: {short.sum()}  long>60m: {longt.sum()})")
    print()
    print(f"  {'P90 source':<22}{'overall':>9}{'short':>9}{'long':>9}   (target 0.90)")
    print("  " + "-" * 56)

    def cov(q, m):
        return float(np.mean(yte[m] <= q[m]))
    for name, q in [("GBM model P90", model_p90), ("F2 empirical P90", f2_p90)]:
        print(f"  {name:<22}{cov(q,np.ones(len(yte),bool)):>9.3f}"
              f"{cov(q,short):>9.3f}{cov(q,longt):>9.3f}")

    # ---- Per-type coverage for the F2 empirical P90 (where it matters) ----
    print()
    print("  F2 empirical P90 coverage by event type (test):")
    print(f"    {'event_cause':<18}{'n_test':>7}{'train_p90':>11}{'coverage':>10}")
    rows = []
    for c in te["event_cause"].unique():
        m = te["event_cause"].values == c
        if m.sum() < 5:
            continue
        rows.append((c, int(m.sum()), train_p90.get(c, pooled_p90), float(np.mean(yte[m] <= f2_p90[m]))))
    for c, n, p, cv in sorted(rows, key=lambda r: -r[2]):
        print(f"    {c:<18}{n:>7}{p:>11.0f}{cv:>10.3f}")

    # ---- Verdict ----
    g_all, f_all = cov(model_p90, np.ones(len(yte), bool)), cov(f2_p90, np.ones(len(yte), bool))
    g_long, f_long = cov(model_p90, longt), cov(f2_p90, longt)
    print("=" * 74)
    print(" VERDICT")
    print("=" * 74)
    print(f"  overall : GBM {g_all:.1%}  vs  F2 {f_all:.1%}   (target 90%)")
    print(f"  long    : GBM {g_long:.1%}  vs  F2 {f_long:.1%}   <- the events where buffer matters")
    better = "F2 empirical P90" if abs(f_all - 0.9) < abs(g_all - 0.9) else "GBM P90"
    print(f"  -> adopt: {better} as the dispatch buffer (better calibrated, esp. in the tail)")


if __name__ == "__main__":
    main()
