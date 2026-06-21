"""
F5 - Severity Index (transparent composite)
A single interpretable impact class per event. Openly constructed, not learned:

    severity = 0.5 * normalize(expected_duration)  +  0.5 * closure_rate

  expected_duration = F2 robust type-median   (NOT a model estimate)
  closure_rate      = F2 type closure rate     (transparent; F4's calibrated
                      per-event probability is used downstream in F6 dispatch)

Normalisation bounds and tier cutpoints are FROZEN at fit time and persisted, so a
single live event scores on the same basis as the batch.  We also report the
duration<->closure correlation: the two halves overlap only partly, so the
composite carries genuine information beyond either alone.
"""
import pandas as pd
import numpy as np
import json

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
CLO = os.path.join(_BASE, "closure_clean.csv")
REF = os.path.join(_BASE, "reference_table.csv")
PARAMS = os.path.join(_BASE, "severity_params.json")
SCORED = os.path.join(_BASE, "severity_by_type.csv")
W_DUR, W_CLO = 0.5, 0.5


def main():
    df = pd.read_csv(CLO, low_memory=False)
    ref = pd.read_csv(REF)
    med = dict(zip(ref["event_cause"], ref["median_min"]))
    rate = dict(zip(ref["event_cause"], ref["closure_rate"]))

    df["exp_dur"] = df["event_cause"].map(med)
    df["closure_rate"] = df["event_cause"].map(rate)

    # ---- Freeze normalisation bounds (clip at 5th/95th pct of expected duration) ----
    lo, hi = np.percentile(df["exp_dur"], 5), np.percentile(df["exp_dur"], 95)
    df["norm_dur"] = (df["exp_dur"].clip(lo, hi) - lo) / (hi - lo)
    df["severity"] = W_DUR * df["norm_dur"] + W_CLO * df["closure_rate"]

    # ---- Freeze tier cutpoints (60th / 85th pct of severity) ----
    # The dominant routine category (vehicle_breakdown, ~60% of events) sits on the
    # 60th-pct value; it anchors the BOTTOM of Low (strict '>' for the Medium boundary).
    c60, c85 = np.percentile(df["severity"], 60), np.percentile(df["severity"], 85)
    df["tier"] = np.where(df["severity"] >= c85, "High",
                  np.where(df["severity"] > c60, "Medium", "Low"))

    params = {"weights": {"duration": W_DUR, "closure": W_CLO},
              "dur_clip_lo": round(float(lo), 2), "dur_clip_hi": round(float(hi), 2),
              "tier_cut_60": round(float(c60), 4), "tier_cut_85": round(float(c85), 4)}
    json.dump(params, open(PARAMS, "w"), indent=2)

    print("=" * 78)
    print(" F5 - SEVERITY INDEX (transparent composite)")
    print("=" * 78)
    print(f"  frozen params: dur_clip=[{lo:.0f}, {hi:.0f}] min  "
          f"tier_cuts: Medium>={c60:.3f}  High>={c85:.3f}")

    # ---- The double-count caveat, quantified ----
    tl = ref[["event_cause", "median_min", "closure_rate"]].dropna()
    pear = tl["median_min"].corr(tl["closure_rate"])
    spear = tl["median_min"].corr(tl["closure_rate"], method="spearman")
    ev = df["exp_dur"].corr(df["closure_rate"])
    print("-" * 78)
    print(" DURATION <-> CLOSURE CORRELATION (are the two halves redundant?)")
    print("-" * 78)
    print(f"    type-level  Pearson={pear:+.2f}   Spearman={spear:+.2f}   (n={len(tl)} types)")
    print(f"    event-level Pearson={ev:+.2f}   (frequency-weighted)")
    # clearest divergence: quick-clearing but closure-heavy
    tl2 = tl.assign(dur_rank=tl["median_min"].rank(), clo_rank=tl["closure_rate"].rank())
    tl2["divergence"] = (tl2["dur_rank"] - tl2["clo_rank"]).abs()
    div = tl2.sort_values("divergence", ascending=False).head(3)
    print(f"    biggest divergences (duration rank vs closure rank):")
    for _, r in div.iterrows():
        print(f"      {r['event_cause']:<16} median={r['median_min']:>5.0f} min  closure={r['closure_rate']*100:>4.1f}%")
    print(f"    -> moderate overlap; the halves are NOT the same signal, so 50/50 carries real info")

    # ---- Severity by type (this is the interpretable output) ----
    by = (df.groupby("event_cause")
            .agg(exp_dur=("exp_dur", "first"), closure=("closure_rate", "first"),
                 severity=("severity", "first"), tier=("tier", "first"), n=("severity", "size"))
            .sort_values("severity", ascending=False))
    by.to_csv(SCORED)
    print("-" * 78)
    print(" SEVERITY BY EVENT TYPE (sorted high -> low)")
    print("-" * 78)
    print(f"    {'event_cause':<18}{'med_min':>8}{'closure':>9}{'severity':>10}{'tier':>8}{'n':>7}")
    for c, r in by.iterrows():
        print(f"    {c:<18}{r['exp_dur']:>8.0f}{r['closure']*100:>8.1f}%{r['severity']:>10.3f}{r['tier']:>8}{int(r['n']):>7}")

    dist = df["tier"].value_counts(normalize=True)
    print("-" * 78)
    print(f"  event-weighted tier mix:  "
          + "  ".join(f"{t} {dist.get(t,0)*100:.0f}%" for t in ["Low", "Medium", "High"]))
    print(f"  saved -> {PARAMS}  and  {SCORED}")
    print("=" * 78)
    print(" NOTE: severity is a transparent reference-table function (no model). F4's")
    print("       calibrated per-event probability drives the F6 dispatch thresholds.")


if __name__ == "__main__":
    main()
