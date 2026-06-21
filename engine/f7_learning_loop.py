"""
F7 - Post-Event Learning Loop  (solves the 'no post-event learning' pain point)

Two products of deliberately different evidential strength:

  7.1 OUTLIER REVIEW QUEUE (defensible) - events that ran past their peer cohort's
      (event_cause x on/off-corridor) IQR upper fence (Q3 + 1.5*IQR, min cohort 8).
      Genuine anomalies ranked by excess vs cohort median; near-cap tickets (likely
      left open, not real clearance) sort to the bottom.  Never auto-diagnoses blame.

  7.2 CORRIDOR RANKING (hypothesis-only) - volume-normalised outlier rate per corridor
      with Wilson CIs.  Reported as directional: on current data, corridors are NOT
      statistically distinguishable from the global rate.  Sharpens as data accrues.
"""
import pandas as pd
import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
DUR = os.path.join(_BASE, "duration_clean.csv")
CLO = os.path.join(_BASE, "closure_clean.csv")
OUT_Q = os.path.join(_BASE, "outlier_queue.csv")
OUT_C = os.path.join(_BASE, "corridor_ranking.csv")
MIN_COHORT = 8
ARTIFACT_MIN = 360    # >6h "clearance" is implausible for active road-clearance ->
                      # almost certainly a left-open ticket (lifecycle), not a real anomaly


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    df = pd.read_csv(DUR, low_memory=False)
    df["cohort"] = df["event_cause"] + " | " + np.where(df["on_corridor"], "on-corr", "off-corr")

    grp = df.groupby("cohort")["duration_min"]
    df["cohort_n"] = grp.transform("size")
    df["cohort_median"] = grp.transform("median")
    df["fence"] = grp.transform(
        lambda s: (s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25)))
        if len(s) >= MIN_COHORT else np.nan)

    df["outlier"] = (df["cohort_n"] >= MIN_COHORT) & (df["duration_min"] > df["fence"])
    df["likely_artifact"] = df["duration_min"] > ARTIFACT_MIN
    df["excess_ratio"] = df["duration_min"] / df["cohort_median"]

    flaggable = (df["cohort_n"] >= MIN_COHORT).sum()
    rate = df["outlier"].mean()

    print("=" * 78)
    print(" F7 - POST-EVENT LEARNING LOOP")
    print("=" * 78)
    print(f"  reviewed {len(df)} clean-clearance events | flaggable (cohort>={MIN_COHORT}): {flaggable}")
    print(f"  outlier rate: {rate:.1%}  ({int(df['outlier'].sum())} events past their cohort fence)")

    # ---- 7.1 Outlier review queue ----
    q = df[df["outlier"]].copy()
    q["kind"] = np.where(q["likely_artifact"], "ticket_artifact", "genuine")
    q = q.sort_values(["likely_artifact", "excess_ratio"], ascending=[True, False]).reset_index(drop=True)
    q["rank"] = q.index + 1
    qcols = ["rank", "id", "event_cause", "on_corridor", "duration_min",
             "cohort_median", "fence", "excess_ratio", "kind"]
    q[qcols].round(2).to_csv(OUT_Q, index=False)
    n_gen = (q["kind"] == "genuine").sum()
    n_art = (q["kind"] == "ticket_artifact").sum()

    print("-" * 78)
    print(f" 7.1 OUTLIER REVIEW QUEUE: {n_gen} genuine (<=6h) + {n_art} ticket-artifact = {len(q)} flagged")
    print("-" * 78)
    print(f"    {'#':>3}{'event_cause':<18}{'corr':>5}{'dur_min':>9}{'cohort_med':>11}{'xs':>7}  kind")
    for _, r in q.head(10).iterrows():
        print(f"    {int(r['rank']):>3}{r['event_cause']:<18}{('on' if r['on_corridor'] else 'off'):>5}"
              f"{r['duration_min']:>9.0f}{r['cohort_median']:>11.0f}{r['excess_ratio']:>6.1f}x  {r['kind']}")
    print("    (xs = duration / cohort median;  flags for human review only - never auto-blame)")

    # ---- lifecycle backlog (the F6 review-flag population) for context ----
    clo = pd.read_csv(CLO, low_memory=False)
    backlog = int((clo["near_24h_cap"] == True).sum())
    print(f"    + lifecycle backlog: {backlog} events capped at >24h (left-open tickets, "
          f"feed the same review path)")

    # ---- 7.2 Corridor ranking (hypothesis-only) ----
    g = df.groupby("corridor").agg(n=("outlier", "size"), outliers=("outlier", "sum"))
    g = g[g["n"] >= 20].copy()                       # stability floor
    g["rate"] = g["outliers"] / g["n"]
    ci = g.apply(lambda r: wilson_ci(r["outliers"], r["n"]), axis=1)
    g["lo95"], g["hi95"] = [c[0] for c in ci], [c[1] for c in ci]
    g["distinguishable"] = (g["lo95"] > rate) | (g["hi95"] < rate)
    g = g.sort_values("rate", ascending=False)
    g.round(3).to_csv(OUT_C)

    print("-" * 78)
    print(f" 7.2 CORRIDOR RANKING (hypothesis-only; global outlier rate = {rate:.1%})")
    print("-" * 78)
    print(f"    {'corridor':<22}{'n':>6}{'outlier_rate':>14}{'  95% Wilson CI':>20}{'  distinct?':>12}")
    for c, r in g.head(8).iterrows():
        ci_s = f"[{r['lo95']:.2f}, {r['hi95']:.2f}]"
        print(f"    {str(c)[:21]:<22}{int(r['n']):>6}{r['rate']*100:>13.1f}%{ci_s:>20}"
              f"{('YES' if r['distinguishable'] else 'no'):>12}")
    n_distinct = int(g["distinguishable"].sum())
    print("-" * 78)
    print(" VERDICT")
    print("=" * 78)
    print(f"  Review queue: {n_gen} genuine anomalies surfaced for human review, ranked by excess.")
    print(f"  Corridor ranking: {n_distinct}/{len(g)} corridors statistically distinguishable from "
          f"the {rate:.1%} global rate")
    print(f"    -> {'directional only; no corridor is a proven hotspot yet (honest, sharpens with data)' if n_distinct==0 else 'see flagged corridors'}")
    print(f"  saved -> {OUT_Q}  and  {OUT_C}")


if __name__ == "__main__":
    main()
