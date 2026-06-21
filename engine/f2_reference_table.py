"""
F2 - Per-Type Reference Table  (the primary predictor)
Flipkart GRiD Round 2 | Event-Impact & Resource Recommendation Engine

One row per event cause. Operator-readable, and the anchor every other module
leans on. Reads the canonical Step 0 datasets only.

  duration stats  <- duration_clean.csv  (real-close events in (0, 24h])
  closure stats   <- closure_clean.csv    (full population)

Outputs reference_table.csv + a printed table, and exposes reference_lookup().
"""
import pandas as pd
import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
DUR = os.path.join(_BASE, "duration_clean.csv")
CLO = os.path.join(_BASE, "closure_clean.csv")
OUT = os.path.join(_BASE, "reference_table.csv")

RELIABLE_MIN = 30      # n_dur >= 30  -> use own stats
DIRECTIONAL_MIN = 10   # 10..29       -> own, flagged uncertain
# n_dur < 10            -> sparse, duration falls back to pooled global


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def decontaminated_p90(series):
    """Remove within-type IQR outliers (Q3 + 1.5*IQR), then take P90.
    Returns (p90_clean, stale_removed)."""
    s = series.dropna()
    if len(s) == 0:
        return (np.nan, 0)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    kept = s[s <= fence]
    if len(kept) == 0:
        kept = s
    return (kept.quantile(0.90), int(len(s) - len(kept)))


def tier_for(n_dur):
    if n_dur >= RELIABLE_MIN:
        return "reliable"
    if n_dur >= DIRECTIONAL_MIN:
        return "directional"
    return "sparse"


def build_reference_table(dur_df, clo_df):
    # Pooled global fallback (from all clean durations)
    pooled_median = dur_df["duration_min"].median()
    pooled_p90, _ = decontaminated_p90(dur_df["duration_min"])

    rows = []
    for cause in sorted(clo_df["event_cause"].unique()):
        clo_sub = clo_df[clo_df["event_cause"] == cause]
        n_full = len(clo_sub)
        k = int(clo_sub["closure_flag"].sum())
        closure_rate = k / n_full
        w_lo, w_hi = wilson_ci(k, n_full)

        dur_sub = dur_df.loc[dur_df["event_cause"] == cause, "duration_min"]
        n_dur = int(dur_sub.notna().sum())
        tier = tier_for(n_dur)

        if n_dur >= DIRECTIONAL_MIN:           # own stats
            median = dur_sub.median()
            p90, stale = decontaminated_p90(dur_sub)
            dur_source = "own"
        else:                                   # sparse -> pooled fallback
            median, p90, stale = pooled_median, pooled_p90, 0
            dur_source = "pooled"

        rows.append({
            "event_cause": cause,
            "n_dur": n_dur,
            "n_full": n_full,
            "reliability": tier,
            "median_min": round(median, 1),
            "p90_min": round(p90, 1),
            "stale_removed": stale,
            "dur_source": dur_source,
            "closure_rate": round(closure_rate, 3),
            "closure_lo95": round(w_lo, 3),
            "closure_hi95": round(w_hi, 3),
        })

    tbl = pd.DataFrame(rows).sort_values("median_min", ascending=False).reset_index(drop=True)
    tbl.attrs["pooled_median"] = pooled_median
    tbl.attrs["pooled_p90"] = pooled_p90
    tbl.attrs["base_closure"] = clo_df["closure_flag"].mean()
    return tbl


def reference_lookup(tbl, cause):
    """Primary predictor: cause -> expected clearance + closure profile.
    Unknown cause falls back to pooled duration + global closure rate."""
    cause = str(cause).strip().lower()
    hit = tbl[tbl["event_cause"] == cause]
    if len(hit) == 0:
        return {
            "event_cause": cause, "reliability": "unknown",
            "median_min": round(tbl.attrs["pooled_median"], 1),
            "p90_min": round(tbl.attrs["pooled_p90"], 1),
            "closure_rate": round(tbl.attrs["base_closure"], 3),
            "dur_source": "pooled",
        }
    return hit.iloc[0].to_dict()


def main():
    dur_df = pd.read_csv(DUR, low_memory=False)
    clo_df = pd.read_csv(CLO, low_memory=False)
    tbl = build_reference_table(dur_df, clo_df)
    tbl.to_csv(OUT, index=False)

    print("=" * 92)
    print(" F2 - PER-TYPE REFERENCE TABLE   (sorted by median clearance, descending)")
    print("=" * 92)
    hdr = (f"{'event_cause':<18}{'n_dur':>6}{'n_full':>7}{'tier':>13}"
           f"{'median':>8}{'p90':>8}{'stale':>6}{'src':>8}"
           f"{'closure':>9}{'  95% Wilson CI':>18}")
    print(hdr)
    print("-" * 92)
    for _, r in tbl.iterrows():
        ci = f"[{r['closure_lo95']:.2f}, {r['closure_hi95']:.2f}]"
        print(f"{r['event_cause']:<18}{r['n_dur']:>6}{r['n_full']:>7}{r['reliability']:>13}"
              f"{r['median_min']:>8.0f}{r['p90_min']:>8.0f}{r['stale_removed']:>6}{r['dur_source']:>8}"
              f"{r['closure_rate']*100:>8.1f}%{ci:>18}")
    print("-" * 92)
    print(f"  pooled fallback (sparse types): median={tbl.attrs['pooled_median']:.1f} min  "
          f"p90={tbl.attrs['pooled_p90']:.1f} min")
    print(f"  tiers: reliable n>={RELIABLE_MIN} (own) | directional {DIRECTIONAL_MIN}-{RELIABLE_MIN-1} "
          f"(own, flagged) | sparse n<{DIRECTIONAL_MIN} (pooled duration, own closure w/ wide CI)")
    print("=" * 92)

    # Demonstrate the lookup the rest of the engine will call
    print("\n  reference_lookup() examples:")
    for c in ["tree_fall", "vehicle_breakdown", "vip_movement", "flash_mob_not_in_data"]:
        r = reference_lookup(tbl, c)
        print(f"    {c:<24} -> median {r['median_min']:>5.0f} min | p90 {r['p90_min']:>5.0f} min | "
              f"closure {r['closure_rate']*100:>4.1f}% | {r['reliability']}/{r['dur_source']}")


if __name__ == "__main__":
    main()
