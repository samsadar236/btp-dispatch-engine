"""
Alert & Dispatch Agent
Flipkart GRiD Round 2 | Event-Impact & Resource Recommendation Engine

When an incident is reported (live, or handed over by the news agent), score it
immediately through the FULL engine and emit an officer-facing alert.

Unlike the news agent (future events -> type-level analogues), a live incident HAS
features, so it uses F4's calibrated per-event probability with the conservative
type-rate floor (deployment risk = max(F4, type rate)), then the F5/F6 dispatch.

It does NOT auto-dispatch or assign a 'nearest unit' - that needs live fleet
positions the log lacks.  It is decision-support, and it logs every scored incident
to the review feed that F7 consumes.
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from f6_dispatch import dispatch_for_row, CONFIG   # reuse the exact dispatch rules (lives in ../engine)

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
F4 = os.path.join(BASE_DIR, "f4_closure_model.pkl")
REF = os.path.join(BASE_DIR, "reference_table.csv")
SEV = os.path.join(BASE_DIR, "severity_by_type.csv")
BUF = os.path.join(BASE_DIR, "conformal_buffer.csv")
REVIEW_LOG = os.path.join(BASE_DIR, "review_log.csv")


class AlertDispatchAgent:
    def __init__(self):
        self.b = pickle.load(open(F4, "rb"))
        ref = pd.read_csv(REF); sev = pd.read_csv(SEV); buf = pd.read_csv(BUF)
        self.median = dict(zip(ref["event_cause"], ref["median_min"]))
        self.rate = dict(zip(ref["event_cause"], ref["closure_rate"]))
        self.p90_ref = dict(zip(ref["event_cause"], ref["p90_min"]))
        self.tier = dict(zip(sev["event_cause"], sev["tier"]))
        self.buf_p90 = dict(zip(buf["event_cause"], buf["conf_p90_min"]))
        self.buf_status = dict(zip(buf["event_cause"], buf["status"]))

    def _p90(self, c):
        return self.buf_p90.get(c, self.p90_ref.get(c, np.nan))

    def _review(self, c):
        if c in self.buf_status:
            return self.buf_status[c] == "review"
        return self.p90_ref.get(c, 0) > CONFIG["lifecycle_p90_cut_min"]

    def score_incident(self, inc):
        """inc: dict with F4 features (event_cause, priority, event_type, veh_type,
        police_station, zone, on_corridor, desc_*) + lat/lon/start_hour/location."""
        cause = inc["event_cause"]
        row = pd.DataFrame([{c: inc.get(c, "Unknown") for c in self.b["cat"]} |
                            {c: float(inc.get(c, 0)) for c in self.b["num"]}])
        Xc = self.b["encoder"].transform(row[self.b["cat"]].astype(str))
        X = np.concatenate([Xc, row[self.b["num"]].values], axis=1)
        f4 = float(self.b["model"].predict_proba(X)[0, 1])
        type_rate = self.rate.get(cause, f4)
        closure = max(f4, type_rate)                      # deployment-risk floor (== F6)

        rec = dispatch_for_row(cause, closure, self.tier.get(cause, "Medium"),
                               self.median.get(cause, np.nan), self._p90(cause),
                               self._review(cause), inc.get("start_hour"))
        rec.update({"location": inc.get("location"), "lat": inc.get("lat"),
                    "lon": inc.get("lon"), "f4_risk": round(f4, 3),
                    "type_closure_rate": round(type_rate, 3)})
        return rec

    def format_alert(self, rec):
        a = [f"INCIDENT ALERT - {rec['event_cause'].replace('_',' ').title()}",
             f"  Location: {rec.get('location','(coords only)')}  [{rec.get('lat')}, {rec.get('lon')}]",
             f"  Severity: {rec['severity_tier'].upper()} | Closure risk: {rec['closure_risk']:.0%} "
             f"(F4 {rec['f4_risk']:.0%} / type {rec['type_closure_rate']:.0%})",
             f"  Expected clearance: {rec['expected_clearance_min']} min  |  "
             f"hold ~{rec['resource_hold_min']/60:.1f}h",
             f"  ACTION: Deploy {rec['recommended_personnel']} officer(s)"
             + (" | Barricade" if rec["barricade"] else "")
             + (" | Diversion" if rec["diversion"] else "")]
        for n in rec["notes"]:
            a.append(f"  ! {n}")
        return "\n".join(a)

    def log_to_review(self, rec):
        """Append the scored incident to the feed F7 consumes (institutional memory)."""
        keep = {k: rec.get(k) for k in ["event_cause", "severity_tier", "closure_risk",
                "expected_clearance_min", "recommended_personnel", "barricade",
                "diversion", "review_flag", "location"]}
        pd.DataFrame([keep]).to_csv(REVIEW_LOG, mode="a", header=not _exists(REVIEW_LOG), index=False)


def _exists(p):
    import os
    return os.path.exists(p)


def main():
    import os
    if os.path.exists(REVIEW_LOG):
        os.remove(REVIEW_LOG)
    agent = AlertDispatchAgent()

    incidents = [
        {"event_cause": "tree_fall", "priority": "High", "event_type": "unplanned",
         "veh_type": "not_applicable", "police_station": "Sadashivanagar", "zone": "East",
         "on_corridor": 1, "desc_highway": 0, "desc_multiple": 0, "desc_signal": 0,
         "location": "Bellary Road, near Mekhri Circle", "lat": 13.01, "lon": 77.58, "start_hour": 14},
        {"event_cause": "vehicle_breakdown", "priority": "Low", "event_type": "unplanned",
         "veh_type": "bmtc_bus", "police_station": "Madiwala", "zone": "South",
         "on_corridor": 1, "desc_highway": 0, "desc_multiple": 0, "desc_signal": 0,
         "location": "Hosur Road, Silk Board", "lat": 12.92, "lon": 77.62, "start_hour": 18},
        {"event_cause": "accident", "priority": "High", "event_type": "unplanned",
         "veh_type": "heavy_vehicle", "police_station": "KR Puram", "zone": "East",
         "on_corridor": 1, "desc_highway": 1, "desc_multiple": 1, "desc_signal": 0,
         "location": "Old Madras Road flyover", "lat": 13.00, "lon": 77.68, "start_hour": 2},
        {"event_cause": "vip_movement", "priority": "High", "event_type": "planned",
         "veh_type": "not_applicable", "police_station": "Vidhana Soudha", "zone": "Central",
         "on_corridor": 1, "desc_highway": 0, "desc_multiple": 0, "desc_signal": 0,
         "location": "Vidhana Soudha approach", "lat": 12.98, "lon": 77.59, "start_hour": 10},
    ]

    print("=" * 78)
    print(" ALERT & DISPATCH AGENT  -  live incident -> immediate scored alert")
    print("=" * 78)
    print("  cascade: report -> score (F2-F6, F4 per-event) -> alert -> log to F7 feed")
    print("  (decision-support only: no auto-dispatch, no 'nearest unit' - needs live fleet data)\n")
    for inc in incidents:
        rec = agent.score_incident(inc)
        agent.log_to_review(rec)
        print(agent.format_alert(rec))
        print()
    print("=" * 78)
    print(f"  logged {len(incidents)} scored incidents -> {REVIEW_LOG} (feeds F7 learning loop)")


if __name__ == "__main__":
    main()
