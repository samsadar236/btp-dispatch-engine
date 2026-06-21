"""
F6 - Dispatch Recommendation (operating-point rules)
Flipkart GRiD Round 2 | Event-Impact & Resource Recommendation Engine

Converts the engine outputs into a per-event dispatch recommendation.  There is no
manpower ground truth in the data, so this is an openly-configurable rules scaffold
(every constant in CONFIG) - but the barricade/diversion thresholds are reported as
OPERATING POINTS on F4's calibrated probability (deploy-rate / recall / precision),
not magic numbers.

Inputs:  closure_clean (events) + reference_table (F2) + F4 model (calibrated prob)
         + severity_by_type (F5) + conformal_buffer (F3 buffer & calibrated/review)
Output:  a MAP-AGNOSTIC dispatch record per event, with a `location_name` slot left
         null for the Mappls viz layer to fill downstream.  Core never imports a map.
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split

# ---- portability: defaults to the folder this script lives in ----
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
CLO = os.path.join(BASE_DIR, "closure_clean.csv")
REF = os.path.join(BASE_DIR, "reference_table.csv")
SEV = os.path.join(BASE_DIR, "severity_by_type.csv")
BUF = os.path.join(BASE_DIR, "conformal_buffer.csv")
F4 = os.path.join(BASE_DIR, "f4_closure_model.pkl")
OUT_CSV = os.path.join(BASE_DIR, "dispatch_schema.csv")
OUT_JSON = os.path.join(BASE_DIR, "dispatch_records_sample.json")

CONFIG = {
    "barricade_threshold": 0.30,
    "diversion_threshold": 0.50,
    "personnel_by_tier": {"Low": 1, "Medium": 2, "High": 3},
    "high_closure_bump_threshold": 0.50,   # +1 officer above this closure risk
    "night_hours": [22, 23, 0, 1, 2, 3, 4, 5],
    "lifecycle_p90_cut_min": 360,          # buffer > 6h => ticket-lifecycle, dispatch on median
}


def dispatch_for_row(cause, closure_risk, tier, median, p90, review, hour):
    """Map a scored event to a dispatch recommendation record (map-agnostic)."""
    personnel = CONFIG["personnel_by_tier"].get(tier, 2)
    if closure_risk >= CONFIG["high_closure_bump_threshold"]:
        personnel += 1
    barricade = closure_risk >= CONFIG["barricade_threshold"]
    diversion = (closure_risk >= CONFIG["diversion_threshold"]) or (tier == "High")
    night = hour in CONFIG["night_hours"] if pd.notna(hour) else False
    hold = median if review else p90               # don't hold for a ticket-lifecycle tail

    notes = []
    if review:
        notes.append("duration tail is ticket-lifecycle; dispatch on median, route to F7 review")
    if night:
        notes.append("night shift - reduced staffing; verify unit availability")
    return {
        "event_cause": cause,
        "severity_tier": tier,
        "closure_risk": round(float(closure_risk), 3),
        "expected_clearance_min": int(round(median)),
        "p90_buffer_min": int(round(p90)),
        "recommended_personnel": int(personnel),
        "barricade": bool(barricade),
        "diversion": bool(diversion),
        "resource_hold_min": int(round(hold)),
        "review_flag": bool(review),
        "night_flag": bool(night),
        "notes": notes,
        "location_name": None,          # <-- slot for Mappls reverse-geocode (viz layer)
    }


def operating_points(y, p):
    print("-" * 72)
    print(" OPERATING POINTS on deployment risk = max(F4 calibrated prob, type rate)")
    print("-" * 72)
    print(f"    {'threshold':>10}{'deploy_rate':>13}{'recall':>9}{'precision':>11}   note")
    for t in [0.20, 0.30, 0.40, 0.50]:
        flag = p >= t
        deploy = float(flag.mean())
        rec = float((flag & (y == 1)).sum() / max(1, (y == 1).sum()))
        prec = float((flag & (y == 1)).sum() / max(1, flag.sum()))
        tag = "<- BARRICADE" if abs(t - CONFIG["barricade_threshold"]) < 1e-9 else \
              ("<- DIVERSION" if abs(t - CONFIG["diversion_threshold"]) < 1e-9 else "")
        print(f"    {t:>10.2f}{deploy:>13.1%}{rec:>9.1%}{prec:>11.1%}   {tag}")
    print("    (deploy_rate = % of events flagged; recall = % of true closures caught;")
    print("     precision = % of flagged events that truly close)")


def main():
    df = pd.read_csv(CLO, low_memory=False)
    ref = pd.read_csv(REF); sev = pd.read_csv(SEV); buf = pd.read_csv(BUF)

    med = dict(zip(ref["event_cause"], ref["median_min"]))
    crate = dict(zip(ref["event_cause"], ref["closure_rate"]))
    p90_ref = dict(zip(ref["event_cause"], ref["p90_min"]))
    tier = dict(zip(sev["event_cause"], sev["tier"]))
    buf_hold = dict(zip(buf["event_cause"], buf["dispatch_buffer_min"]))
    buf_p90 = dict(zip(buf["event_cause"], buf["conf_p90_min"]))
    buf_status = dict(zip(buf["event_cause"], buf["status"]))

    def p90_of(c):
        return buf_p90.get(c, p90_ref.get(c, np.nan))
    def review_of(c):
        if c in buf_status:
            return buf_status[c] == "review"
        return p90_ref.get(c, 0) > CONFIG["lifecycle_p90_cut_min"]

    # ---- F4 calibrated closure probability for every event ----
    with open(F4, "rb") as f:
        b = pickle.load(f)
    for c in b["num"]:
        df[c] = df[c].astype(float)
    Xc = b["encoder"].transform(df[b["cat"]].astype(str))
    X = np.concatenate([Xc, df[b["num"]].values], axis=1)
    df["f4_risk"] = b["model"].predict_proba(X)[:, 1]
    df["type_rate"] = df["event_cause"].map(crate).fillna(df["f4_risk"])
    # Deployment risk: F4 can RAISE caution via spatial signal, never lower it below the
    # historical type rate (protects rare high-closure types like VIP from noisy per-event est.)
    df["closure_risk"] = np.maximum(df["f4_risk"], df["type_rate"])

    print("=" * 72)
    print(" F6 - DISPATCH RECOMMENDATION ENGINE")
    print("=" * 72)
    print(f"  scored {len(df)} events | thresholds: barricade>={CONFIG['barricade_threshold']} "
          f"diversion>={CONFIG['diversion_threshold']} (configurable)")

    # ---- Assemble the map-agnostic dispatch schema for every event ----
    records = []
    for _, r in df.iterrows():
        c = r["event_cause"]
        rec = dispatch_for_row(c, r["closure_risk"], tier.get(c, "Medium"),
                               med.get(c, np.nan), p90_of(c), review_of(c), r["start_hour"])
        rec["event_id"] = r["id"]
        rec["lat"] = r["latitude"]; rec["lon"] = r["longitude"]
        rec["f4_risk"] = round(float(r["f4_risk"]), 3)                    # F4 spatial estimate
        rec["type_closure_rate"] = round(float(crate.get(c, np.nan)), 3)  # F2 historical anchor
        records.append(rec)
    out = pd.DataFrame(records)
    cols = ["event_id", "lat", "lon", "event_cause", "severity_tier", "closure_risk",
            "f4_risk", "type_closure_rate", "expected_clearance_min", "p90_buffer_min",
            "recommended_personnel", "barricade", "diversion", "resource_hold_min",
            "review_flag", "night_flag", "location_name", "notes"]
    out[cols].to_csv(OUT_CSV, index=False)

    # ---- Deployment summary ----
    print(f"  barricade recommended : {out['barricade'].mean():.1%} of events")
    print(f"  diversion recommended : {out['diversion'].mean():.1%} of events")
    print(f"  review-flagged        : {out['review_flag'].mean():.1%} of events")
    print(f"  personnel mix         : "
          + "  ".join(f"{n} officer(s) {100*(out['recommended_personnel']==n).mean():.0f}%"
                      for n in sorted(out["recommended_personnel"].unique())))

    # ---- Operating points (held-out, honest) ----
    y = (df["requires_road_closure"] == True).astype(int).values
    _, Xte, _, yte, _, pte = train_test_split(
        X, y, df["closure_risk"].values, test_size=0.25, random_state=42, stratify=y)
    operating_points(yte, pte)

    # ---- Example recommendations (human-readable) ----
    print("-" * 72)
    print(" EXAMPLE RECOMMENDATIONS")
    print("-" * 72)
    examples = {}
    for c in ["construction", "tree_fall", "vip_movement", "vehicle_breakdown", "pot_holes"]:
        sub = out[out["event_cause"] == c]
        if len(sub):
            examples[c] = sub.iloc[0]
    for c, r in examples.items():
        hold_h = r["resource_hold_min"] / 60
        print(f"\n  {c.replace('_',' ').title()}  (event {r['event_id']})")
        print(f"    Severity: {r['severity_tier'].upper()} | Deploy risk: {r['closure_risk']:.0%} "
              f"(max of F4 {r['f4_risk']:.0%}, type {r['type_closure_rate']:.0%}) | "
              f"Clearance: {r['expected_clearance_min']} min")
        print(f"    -> Deploy {r['recommended_personnel']} officer(s)")
        print(f"    -> Barricade: {'YES' if r['barricade'] else 'no'} "
              f"(risk {r['closure_risk']:.0%} vs threshold {CONFIG['barricade_threshold']:.0%})")
        print(f"    -> Diversion: {'YES' if r['diversion'] else 'no'}")
        print(f"    -> Hold resources: ~{hold_h:.1f}h "
              f"({'operational median - review type' if r['review_flag'] else 'conformal P90'})")
        for n in r["notes"]:
            print(f"    ! {n}")

    json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
               for r in [dict(examples[c]) for c in examples]],
              open(OUT_JSON, "w"), indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"  saved dispatch schema -> {OUT_CSV}  ({len(out)} records, map-agnostic)")
    print(f"  saved examples        -> {OUT_JSON}")
    print("  NOTE: 'location_name' is null by design - the Mappls viz layer fills it via")
    print("        reverse-geocode. The engine has no map dependency. 'Nearest unit' is")
    print("        intentionally not assigned (needs live fleet positions the log lacks).")


if __name__ == "__main__":
    main()
