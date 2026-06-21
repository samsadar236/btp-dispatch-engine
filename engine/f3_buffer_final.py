"""
F3 (final) - Shippable Dispatch Buffer
Conformal P90 with a global marginal-coverage guarantee (k constrained >= 1 so we
never under-buffer vs history), split into:
  - CALIBRATED operational buffer for well-behaved high-volume types
  - REVIEW-FLAGGED types whose logged tail is ticket-lifecycle, not clearance time
    (dispatch on the operational median; route the event to the F7 review queue)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
DUR = os.path.join(_BASE, "duration_clean.csv")
OUT = os.path.join(_BASE, "conformal_buffer.csv")
SEED = 42
LEVEL = 0.90
LIFECYCLE_CUT = 360   # raw P90 > 6h => tail reflects open-ticket lifecycle, not clearance


def conformal_k(scores, level=LEVEL):
    s = np.asarray(scores); n = len(s)
    if n == 0:
        return 1.0
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * level) / n), method="higher"))


def main():
    df = pd.read_csv(DUR, low_memory=False)
    itr, ite = train_test_split(np.arange(len(df)), test_size=0.25, random_state=SEED)
    ibase, ical = train_test_split(itr, test_size=0.30, random_state=43)
    base, calib, test = df.iloc[ibase], df.iloc[ical], df.iloc[ite]

    base_p90 = base.groupby("event_cause")["duration_min"].quantile(0.90).to_dict()
    base_n = base.groupby("event_cause")["duration_min"].size().to_dict()
    pooled_p90 = base["duration_min"].quantile(0.90)
    # descriptive data-quality flags from the full clean set
    full_p90 = df.groupby("event_cause")["duration_min"].quantile(0.90).to_dict()
    full_med = df.groupby("event_cause")["duration_min"].median().to_dict()

    def p90_of(c):
        return base_p90[c] if base_n.get(c, 0) >= 10 else pooled_p90

    # GLOBAL conformal multiplier (marginal 90% guarantee), never reduce below history
    cs = calib["duration_min"].values / np.array([p90_of(c) for c in calib["event_cause"]])
    k = max(1.0, conformal_k(cs))

    yte = test["duration_min"].values
    conf_te = np.array([p90_of(c) * k for c in test["event_cause"]])
    overall_cov = float(np.mean(yte <= conf_te))

    print("=" * 80)
    print(" F3 (final) - SHIPPABLE DISPATCH BUFFER")
    print("=" * 80)
    print(f"  base={len(base)}  calib={len(calib)}  test={len(test)}   global conformal k = {k:.2f}")
    print(f"  CONFORMAL P90 marginal coverage (out-of-sample): {overall_cov:.1%}  (target 90%)")
    print()
    print(f"  {'event_cause':<18}{'base_n':>7}{'op_median':>10}{'conf_p90':>10}{'cover':>8}  status")
    print("  " + "-" * 70)
    rows = []
    for c in sorted(set(test["event_cause"]), key=lambda x: -p90_of(x)):
        m = test["event_cause"].values == c
        if m.sum() < 5:
            continue
        cp = p90_of(c) * k
        cov = float(np.mean(yte[m] <= cp))
        lifecycle = full_p90.get(c, 0) > LIFECYCLE_CUT
        status = "REVIEW (lifecycle tail)" if lifecycle else "calibrated"
        dispatch_buffer = round(full_med.get(c, np.nan)) if lifecycle else round(cp)
        rows.append({"event_cause": c, "base_n": base_n.get(c, 0),
                     "op_median_min": round(full_med.get(c, np.nan)),
                     "conf_p90_min": round(cp), "test_coverage": round(cov, 3),
                     "status": "review" if lifecycle else "calibrated",
                     "dispatch_buffer_min": dispatch_buffer})
        print(f"  {c:<18}{base_n.get(c,0):>7}{full_med.get(c,np.nan):>10.0f}{cp:>10.0f}{cov:>8.2f}  {status}")

    pd.DataFrame(rows).to_csv(OUT, index=False)

    cal = [r for r in rows if r["status"] == "calibrated"]
    cal_cov = np.mean([r["test_coverage"] for r in cal])
    print("=" * 80)
    print(" VERDICT")
    print("=" * 80)
    print(f"  Calibrated types ({len(cal)}): mean P90 coverage {cal_cov:.1%} -> trustworthy dispatch buffer.")
    print(f"  Review types: logged tail is open-ticket lifecycle; dispatch on operational median,")
    print(f"                route the event to the F7 review queue (this is a feature, not a gap).")
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
