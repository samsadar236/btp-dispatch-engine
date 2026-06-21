"""
Step 0 - Canonical Data Cleaning Pass
Flipkart GRiD Round 2 | Event-Impact & Resource Recommendation Engine

Runs FIRST. Produces two canonical datasets + a printed audit log so every
cleaning decision is reproducible and verifiable before any modeling.

Outputs:
  - closure_clean.csv   (full population, for the closure-risk model)
  - duration_clean.csv  (real-close events in (0, 24h], for the timing models)
"""

import pandas as pd
import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
SRC = os.path.join(_BASE, "Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv")
OUT_DIR = _BASE

# Vehicle-related causes: veh_type is meaningful here; elsewhere it is structurally N/A
VEHICLE_CAUSES = {"vehicle_breakdown", "accident"}

# Columns we keep through to the canonical datasets
KEEP = [
    "id", "event_cause", "event_type", "status", "priority",
    "requires_road_closure", "corridor", "veh_type",
    "police_station", "zone", "junction",
    "latitude", "longitude", "description",
    "start_datetime", "closed_datetime",
]


def log(section, msg=""):
    print(f"  [{section:<16}] {msg}")


def main():
    print("=" * 68)
    print(" STEP 0 - CANONICAL CLEANING AUDIT")
    print("=" * 68)

    # ---- 1. LOAD + INTEGRITY -------------------------------------------------
    df = pd.read_csv(SRC, low_memory=False)
    assert df.shape == (8173, 46), f"unexpected shape {df.shape}"
    log("LOAD", f"rows={df.shape[0]}  cols={df.shape[1]}   shape asserted OK")
    n_dup_rows = int(df.duplicated().sum())
    n_dup_ids = int(df["id"].duplicated().sum())
    assert n_dup_rows == 0 and n_dup_ids == 0
    log("INTEGRITY", f"duplicate rows={n_dup_rows}  duplicate ids={n_dup_ids}   OK")

    n0 = len(df)

    # ---- 2. DROP TEST/DEMO + CASE NORMALISE ---------------------------------
    df["event_cause"] = df["event_cause"].astype(str).str.strip().str.lower()
    n_debris = int((df["event_cause"] == "debris").sum())
    is_test = df["event_cause"] == "test_demo"
    n_test = int(is_test.sum())
    df = df[~is_test].copy()
    log("DROP test_demo", f"removed={n_test}  ->  {len(df)} rows")
    log("CASE FIX", f"'Debris'/'debris' merged -> {df['event_cause'].nunique()} distinct causes "
                    f"(debris now n={n_debris})")

    # ---- 3. DATETIMES (local IST, NO tz conversion) + DURATION ---------------
    # Values carry a +00 label but are Bengaluru local time; parsing utc=True and
    # reading the wall-clock as-is recovers the true local hour (verified: 21:00 peak).
    start = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    closed = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)
    df["start_hour"] = start.dt.hour                      # local IST hour (EDA / shift layer only)
    df["duration_min"] = (closed - start).dt.total_seconds() / 60.0
    log("DATETIME", "parsed as local IST (no tz shift); duration = closed - start")

    # ---- 4. CLOSURE POPULATION (full surviving rows) -------------------------
    df["closure_flag"] = df["requires_road_closure"] == True  # set at creation -> trustworthy
    closure_rate = df["closure_flag"].mean()
    log("CLOSURE POP", f"{len(df)} rows | closure rate={closure_rate:.3f} "
                       f"({int(df['closure_flag'].sum())} closures)")

    # ---- 5. DURATION POPULATION ---------------------------------------------
    print("  [DURATION POP    ]")
    has_close = closed.notna()
    rescue = (df["status"] == "closed") & (~has_close)     # fabricated-duration rescue rows
    n_present = int(has_close.sum())
    n_rescue = int(rescue.sum())
    n_neg = int(((df["duration_min"] <= 0) & has_close).sum())
    over_24h = has_close & (df["duration_min"] > 1440)
    n_over = int(over_24h.sum())
    dur_mask = has_close & (df["duration_min"] > 0) & (df["duration_min"] <= 1440)
    print(f"      closed_datetime present        : {n_present}")
    print(f"      rescue (status=closed, no close): {n_rescue}  -> excluded (no genuine close time)")
    print(f"      non-positive duration dropped   : {n_neg}")
    print(f"      > 24h capped (flagged, excluded): {n_over}")
    print(f"      FINAL duration population       : {int(dur_mask.sum())}")

    # near_24h_cap flag preserved on the FULL frame for the F7 learning loop
    df["near_24h_cap"] = over_24h.values

    # ---- 6. MISSING-DATA POLICY ---------------------------------------------
    print("  [MISSING-DATA    ] applying per-column policy:")
    # veh_type: structural N/A off the vehicle causes; Unknown only if missing on a vehicle cause
    veh_null = df["veh_type"].isna()
    on_vehicle = df["event_cause"].isin(VEHICLE_CAUSES)
    df["veh_type"] = np.where(veh_null & ~on_vehicle, "not_applicable",
                       np.where(veh_null & on_vehicle, "unknown", df["veh_type"]))
    print(f"      veh_type      : {int((veh_null & ~on_vehicle).sum())} -> not_applicable, "
          f"{int((veh_null & on_vehicle).sum())} -> unknown")
    # corridor -> on/off binary (police_station stays primary spatial; zone/junction supplementary)
    df["on_corridor"] = (df["corridor"].notna() & (df["corridor"] != "Non-corridor"))
    df["corridor"] = df["corridor"].fillna("Unknown")
    df["zone"] = df["zone"].fillna("Unknown")
    df["junction"] = df["junction"].fillna("Unknown")      # 69% Unknown -> NOT a model feature
    df["priority"] = df["priority"].fillna("Unknown")
    print(f"      corridor      : on/off binary derived (on_corridor); {int((df['corridor']=='Unknown').sum())} null->Unknown")
    print(f"      zone/junction : filled Unknown (junction kept display/supplementary only)")
    print(f"      priority      : {int((df['priority']=='Unknown').sum())} null->Unknown")

    # description text flags (zeroed when missing)
    desc = df["description"].fillna("").astype(str).str.lower()
    df["desc_highway"] = desc.str.contains("highway")
    df["desc_multiple"] = desc.str.contains("multiple")
    df["desc_signal"] = desc.str.contains("signal")
    print(f"      description    : text flags derived (highway/multiple/signal); "
          f"{int(df['description'].isna().sum())} missing -> zeroed")

    # ---- 7. SELECT + EMIT ----------------------------------------------------
    derived = ["start_hour", "closure_flag", "on_corridor", "near_24h_cap",
               "desc_highway", "desc_multiple", "desc_signal", "duration_min"]
    cols = [c for c in KEEP if c in df.columns] + derived

    closure_clean = df[cols].copy()
    duration_clean = df.loc[dur_mask, cols].copy()

    closure_path = f"{OUT_DIR}/closure_clean.csv"
    duration_path = f"{OUT_DIR}/duration_clean.csv"
    closure_clean.to_csv(closure_path, index=False)
    duration_clean.to_csv(duration_path, index=False)

    print("  [OUTPUTS         ]")
    print(f"      closure_clean.csv  : {closure_clean.shape[0]} x {closure_clean.shape[1]}")
    print(f"      duration_clean.csv : {duration_clean.shape[0]} x {duration_clean.shape[1]}")

    # ---- SANITY SNAPSHOT (so we verify the canonical sets before modeling) ----
    print("=" * 68)
    print(" SANITY SNAPSHOT")
    print("=" * 68)
    print(f"  rows in -> out         : {n0}  ->  closure {len(closure_clean)} / duration {len(duration_clean)}")
    print(f"  duration median (min)  : {duration_clean['duration_min'].median():.1f}")
    print(f"  duration p90 (min)     : {duration_clean['duration_min'].quantile(0.90):.1f}")
    print(f"  overall closure rate   : {closure_clean['closure_flag'].mean():.3f}")
    print()
    print("  closure rate by priority (the headline inversion):")
    pr = closure_clean.groupby("priority")["closure_flag"].agg(["mean", "size"])
    for p, row in pr.iterrows():
        print(f"      {p:<8} {row['mean']:.3f}  (n={int(row['size'])})")
    print()
    print("  duration median by event_cause (top 8 by count):")
    g = (duration_clean.groupby("event_cause")["duration_min"]
         .agg(["median", "size"]).sort_values("size", ascending=False).head(8))
    for c, row in g.iterrows():
        print(f"      {c:<18} median={row['median']:6.1f}  n={int(row['size'])}")
    print()
    print("  sub-type split within vehicle_breakdown (the main accuracy lever):")
    vb = duration_clean[duration_clean["event_cause"] == "vehicle_breakdown"]
    vg = vb.groupby("veh_type")["duration_min"].agg(["median", "size"]).sort_values("median", ascending=False)
    for v, row in vg.iterrows():
        if row["size"] >= 15:
            print(f"      {v:<14} median={row['median']:6.1f}  n={int(row['size'])}")


if __name__ == "__main__":
    main()
