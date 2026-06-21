"""
News & Event Watch Agent (spine) - the labeled extension
Flipkart GRiD Round 2 | Event-Impact & Resource Recommendation Engine

Detects UPCOMING Bengaluru events and PRE-SCORES their impact from historical
analogues in the reference table - before they happen.  Directly addresses pain
point 1, "event impact not quantified in advance."

This container has no network, so the two outward-facing pieces are defined as
INTERFACES to wire with keys at deploy:
  * fetch_notices()          - source ingestion (BTP advisory / one RSS feed)
  * extract_event_from_text  - LLM extraction (Claude API, structured prompt)
The SPINE - map a detected event type to an event_cause analogue, then pre-score
through the engine - runs here exactly as it would on LLM-extracted records.

A future event has no exact location features, so pre-scoring uses the TYPE-level
analogue (F2 reference + F5 severity + conformal buffer) - the honest best estimate.
"""
import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
from f6_dispatch import dispatch_for_row, CONFIG   # reuse the exact dispatch rules (lives in ../engine)

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_HERE), "data"))  # ../data
REF = os.path.join(BASE_DIR, "reference_table.csv")
SEV = os.path.join(BASE_DIR, "severity_by_type.csv")
BUF = os.path.join(BASE_DIR, "conformal_buffer.csv")

# Detected event type -> nearest event_cause analogue in the reference table
EVENT_TYPE_MAP = {
    "rally": "protest", "protest": "protest", "march": "protest", "strike": "protest",
    "demonstration": "protest", "bandh": "protest",
    "festival": "public_event", "concert": "public_event", "sports": "public_event",
    "match": "public_event", "fair": "public_event", "exhibition": "public_event",
    "vip": "vip_movement", "motorcade": "vip_movement", "dignitary": "vip_movement",
    "convoy": "vip_movement", "state_visit": "vip_movement",
    "procession": "procession", "parade": "procession", "religious": "procession",
    "construction": "construction", "roadwork": "construction", "digging": "construction",
    "maintenance": "road_conditions",
}


def map_to_cause(event_type):
    return EVENT_TYPE_MAP.get(str(event_type).lower().strip(), "public_event")


def fetch_notices():
    """[SOURCE INTERFACE] At deploy: scrape one reliable feed (BTP traffic advisory
    portal or a single Bengaluru RSS) and return a list of raw notice texts.
    No network here."""
    raise NotImplementedError("Wire to BTP advisory / RSS at deploy.")


def extract_event_from_text(notice_text):
    """[LLM INTERFACE] At deploy: call Claude (claude-sonnet-4-6) with a structured
    extraction prompt returning:
        {event_name, location, event_type, start_time, end_time, expected_scale, confidence}
    No network here, so the spine below runs on the structured record directly."""
    raise NotImplementedError("Wire to Claude API at deploy; structured-extraction prompt.")


class NewsWatchAgent:
    def __init__(self):
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

    def prescore(self, ev):
        """ev: {event_name, location, event_type, start_hour, expected_scale, confidence}.
        Pre-event recommendation from the type-level historical analogue."""
        cause = map_to_cause(ev["event_type"])
        closure = self.rate.get(cause, np.nan)          # historical type rate = the analogue
        rec = dispatch_for_row(cause, closure, self.tier.get(cause, "Medium"),
                               self.median.get(cause, np.nan), self._p90(cause),
                               self._review(cause), ev.get("start_hour"))
        rec.update({"event_name": ev["event_name"], "location": ev["location"],
                    "analogue_cause": cause, "expected_scale": ev.get("expected_scale"),
                    "extraction_confidence": ev.get("confidence"),
                    "closure_basis": "historical type rate (pre-event analogue)"})
        return rec


def main():
    agent = NewsWatchAgent()
    # Structured records as the LLM extractor would return them
    upcoming = [
        {"event_name": "Farmers' rally", "location": "Town Hall, MG Road",
         "event_type": "rally", "start_hour": 11, "expected_scale": "large", "confidence": 0.91},
        {"event_name": "State dignitary motorcade", "location": "Airport Rd -> Vidhana Soudha",
         "event_type": "motorcade", "start_hour": 9, "expected_scale": "medium", "confidence": 0.88},
        {"event_name": "Cricket fixture", "location": "Chinnaswamy Stadium",
         "event_type": "match", "start_hour": 19, "expected_scale": "large", "confidence": 0.95},
        {"event_name": "Flyover repair works", "location": "Hosur Road",
         "event_type": "roadwork", "start_hour": 23, "expected_scale": "medium", "confidence": 0.84},
        {"event_name": "Temple car procession", "location": "Basavanagudi",
         "event_type": "procession", "start_hour": 7, "expected_scale": "medium", "confidence": 0.79},
    ]

    print("=" * 80)
    print(" NEWS & EVENT WATCH AGENT  -  pre-event impact from historical analogues")
    print("=" * 80)
    print("  pipeline: [source] -> [Claude extraction] -> map to event_cause -> pre-score -> push")
    print("  (source + extraction are deploy-time interfaces; spine runs on structured events)\n")

    out = []
    for ev in upcoming:
        r = agent.prescore(ev)
        out.append(r)
        print(f"  UPCOMING: {r['event_name']}  @ {r['location']}  (conf {r['extraction_confidence']:.0%})")
        print(f"    analogue: {r['analogue_cause']}  ->  Severity {r['severity_tier'].upper()} | "
              f"closure ~{r['closure_risk']:.0%} | clearance ~{r['expected_clearance_min']} min")
        actions = []
        actions.append(f"{r['recommended_personnel']} officer(s)")
        if r["barricade"]:
            actions.append("barricade")
        if r["diversion"]:
            actions.append("diversion")
        actions.append(f"hold ~{r['resource_hold_min']/60:.1f}h")
        print(f"    PRE-DEPLOY: " + " | ".join(actions))
        if r["night_flag"]:
            print(f"    ! night shift - verify unit availability")
        print()

    out_path = os.path.join(BASE_DIR, "news_prescored.json")
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print("=" * 80)
    print(f"  pre-scored {len(out)} upcoming events -> {out_path}")
    print("  Honest scope: VIP/public events are ~2% of history, so the analogue is the best")
    print("  achievable pre-estimate - we claim no more precision than the reference table holds.")


if __name__ == "__main__":
    main()
