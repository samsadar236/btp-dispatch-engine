"""
F4 - Closure Risk Model (calibrated probability)
"Does this event need barricades / diversions?"  Trained on the FULL population
(closure flag is set at creation -> no selection bias).  No class weighting, so
probabilities stay calibrated.  We PROVE calibration with a reliability table +
Brier + ECE, exactly as we did for the F3 buffer.

Baseline to beat: the F2 type-lookup closure rate.  We expect only a thin spatial
lift, so the type rate stays primary and the model refines it.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
CLO = os.path.join(_BASE, "closure_clean.csv")
BUNDLE = os.path.join(_BASE, "f4_closure_model.pkl")
CAT = ["event_cause", "priority", "event_type", "veh_type", "police_station", "zone"]
NUM = ["on_corridor", "desc_highway", "desc_multiple", "desc_signal"]
SEED = 42


def reliability(y, p, q=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], q=q, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    ece = float((g["n"] / g["n"].sum() * (g["mean_pred"] - g["observed"]).abs()).sum())
    return g, ece


def main():
    df = pd.read_csv(CLO, low_memory=False)
    for c in NUM:
        df[c] = df[c].astype(float)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xc = enc.fit_transform(df[CAT].astype(str))
    X = np.concatenate([Xc, df[NUM].values], axis=1)
    cat_mask = [True] * len(CAT) + [False] * len(NUM)
    y = df["closure_flag"].astype(int).values
    cause = df["event_cause"].values

    Xtr, Xte, ytr, yte, ctr, cte = train_test_split(
        X, y, cause, test_size=0.25, random_state=SEED, stratify=y)

    print("=" * 76)
    print(" F4 - CLOSURE RISK MODEL  +  CALIBRATION CHECK")
    print("=" * 76)
    print(f"  train={len(Xtr)}  test={len(Xte)}   positive rate: train {ytr.mean():.3f} / test {yte.mean():.3f}")

    # ---- Headline data finding: priority inversion (no model needed) ----
    pr = df.groupby("priority")["closure_flag"].agg(["mean", "size"])
    print(f"  priority inversion (full pop):  "
          + "  ".join(f"{p}={r['mean']:.3f}(n={int(r['size'])})" for p, r in pr.iterrows()))

    # ---- Baseline: F2 type-lookup closure rate (from train) ----
    type_rate = pd.Series(ytr, index=ctr).groupby(level=0).mean().to_dict()
    base_pred = np.array([type_rate.get(c, ytr.mean()) for c in cte])

    # ---- Model: raw and isotonic-calibrated ----
    base = HistGradientBoostingClassifier(
        categorical_features=cat_mask, learning_rate=0.05, max_iter=400,
        max_depth=4, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    base.fit(Xtr, ytr)
    raw_pred = base.predict_proba(Xte)[:, 1]

    cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
    cal.fit(Xtr, ytr)
    cal_pred = cal.predict_proba(Xte)[:, 1]

    print("-" * 76)
    print(" 1) DISCRIMINATION  (AUC, higher is better)")
    print("-" * 76)
    print(f"    type-lookup baseline (F2) : {roc_auc_score(yte, base_pred):.3f}")
    print(f"    model (raw)               : {roc_auc_score(yte, raw_pred):.3f}")
    print(f"    model (isotonic-cal)      : {roc_auc_score(yte, cal_pred):.3f}")

    print("-" * 76)
    print(" 2) CALIBRATION  (Brier + ECE, lower is better; calibration-in-the-large)")
    print("-" * 76)
    for name, p in [("type-lookup", base_pred), ("model raw", raw_pred), ("model isotonic", cal_pred)]:
        _, ece = reliability(yte, p)
        print(f"    {name:<16} Brier={brier_score_loss(yte, p):.4f}  ECE={ece:.4f}  "
              f"mean_pred={p.mean():.3f} vs observed={yte.mean():.3f}")

    # ---- Reliability table for the shipped model ----
    ship_pred, ship_name = (cal_pred, "isotonic") if brier_score_loss(yte, cal_pred) <= brier_score_loss(yte, raw_pred) else (raw_pred, "raw")
    g, ece = reliability(yte, ship_pred)
    print("-" * 76)
    print(f" 3) RELIABILITY TABLE - shipped model ({ship_name})   (mean_pred should ~ observed)")
    print("-" * 76)
    print(f"    {'predicted prob bin':<26}{'n':>6}{'mean_pred':>11}{'observed':>10}")
    for b, r in g.iterrows():
        print(f"    {str(b):<26}{int(r['n']):>6}{r['mean_pred']:>11.3f}{r['observed']:>10.3f}")

    print("=" * 76)
    print(" VERDICT")
    print("=" * 76)
    auc_m, auc_b = roc_auc_score(yte, ship_pred), roc_auc_score(yte, base_pred)
    print(f"  AUC: model {auc_m:.3f} vs type-lookup {auc_b:.3f}  -> lift {auc_m-auc_b:+.3f} "
          f"({'thin spatial refinement; type rate stays primary' if auc_m-auc_b < 0.06 else 'meaningful lift'})")
    print(f"  Calibration: ECE={ece:.4f}, mean_pred={ship_pred.mean():.3f} vs observed={yte.mean():.3f} "
          f"-> {'well calibrated' if ece < 0.03 else 'review'}")

    with open(BUNDLE, "wb") as f:
        pickle.dump({"model": cal if ship_name == "isotonic" else base, "encoder": enc,
                     "cat": CAT, "num": NUM, "cat_mask": cat_mask, "type_rate": type_rate,
                     "calibration": ship_name}, f)
    print(f"  saved -> {BUNDLE}")


if __name__ == "__main__":
    main()
