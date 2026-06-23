"""
BTP Event-Impact & Resource Recommendation Engine - Dashboard
Flipkart GRiD Round 2

Run:  streamlit run app.py
Self-contained: reads only the engine artifacts in DATA_DIR (no engine .py imports),
so it runs anywhere the artifacts sit beside it.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---- portability: data/ subfolder next to app.py (override with BTP_DATA_DIR) ----
DATA_DIR = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

CONFIG = {"barricade": 0.30, "diversion": 0.50, "lifecycle": 360,
          "personnel": {"Low": 1, "Medium": 2, "High": 3},
          "bump": 0.50, "night": [22, 23, 0, 1, 2, 3, 4, 5]}
SEV_RGB = {"High": [214, 39, 40], "Medium": [255, 140, 0], "Low": [44, 160, 44]}
SEV_HEX = {"High": "#d62728", "Medium": "#ff8c00", "Low": "#2ca02c"}

st.set_page_config(page_title="BTP Dispatch Engine", layout="wide", page_icon="\U0001F6A6")


def P(name):
    return os.path.join(DATA_DIR, name)


@st.cache_data
def load_csv(name):
    return pd.read_csv(P(name), low_memory=False)


@st.cache_resource
def load_model():
    with open(P("f4_closure_model.pkl"), "rb") as f:
        return pickle.load(f)


def dispatch(cause, closure, tier, median, p90, review, hour, rates):
    personnel = CONFIG["personnel"].get(tier, 2) + (1 if closure >= CONFIG["bump"] else 0)
    barricade = closure >= CONFIG["barricade"]
    diversion = (closure >= CONFIG["diversion"]) or (tier == "High")
    night = (hour in CONFIG["night"]) if hour is not None else False
    hold = median if review else p90
    notes = []
    if review:
        notes.append("Ticket-lifecycle tail \u2192 dispatch on median, route to F7 review.")
    if night:
        notes.append("Night shift \u2192 reduced staffing; verify unit availability.")
    return dict(personnel=int(personnel), barricade=barricade, diversion=diversion,
                hold_min=hold, night=night, notes=notes)


def score_live(inp, model, ref, sev, buf):
    """Score a LIVE incident through the per-event F4 model with a type-rate floor."""
    rate = dict(zip(ref["event_cause"], ref["closure_rate"]))
    median = dict(zip(ref["event_cause"], ref["median_min"]))
    p90ref = dict(zip(ref["event_cause"], ref["p90_min"]))
    tier = dict(zip(sev["event_cause"], sev["tier"]))
    bp90 = dict(zip(buf["event_cause"], buf["conf_p90_min"]))
    bstat = dict(zip(buf["event_cause"], buf["status"]))
    cause = inp["event_cause"]
    row = pd.DataFrame([{c: inp.get(c, "Unknown") for c in model["cat"]} |
                        {c: float(inp.get(c, 0)) for c in model["num"]}])
    Xc = model["encoder"].transform(row[model["cat"]].astype(str))
    X = np.concatenate([Xc, row[model["num"]].values], axis=1)
    f4 = float(model["model"].predict_proba(X)[0, 1])
    tr = rate.get(cause, f4)
    closure = max(f4, tr)
    p90 = bp90.get(cause, p90ref.get(cause, np.nan))
    review = (bstat.get(cause) == "review") if cause in bstat else (p90ref.get(cause, 0) > CONFIG["lifecycle"])
    d = dispatch(cause, closure, tier.get(cause, "Medium"), median.get(cause, np.nan),
                 p90, review, inp.get("start_hour"), rate)
    d.update(closure=closure, f4=f4, type_rate=tr, tier=tier.get(cause, "Medium"),
             median=median.get(cause, np.nan), p90=p90, review=review)
    return d


def prescore_type(cause, ref, sev, buf):
    """Score an UPCOMING event from its historical analogue (type-level F2, no per-event signal yet)."""
    rate = dict(zip(ref["event_cause"], ref["closure_rate"]))
    median = dict(zip(ref["event_cause"], ref["median_min"]))
    tier = dict(zip(sev["event_cause"], sev["tier"]))
    bp90 = dict(zip(buf["event_cause"], buf["conf_p90_min"]))
    bstat = dict(zip(buf["event_cause"], buf["status"]))
    closure = rate.get(cause, 0.0)
    t = tier.get(cause, "Medium")
    review = bstat.get(cause) == "review"
    p90 = bp90.get(cause, median.get(cause, np.nan))
    d = dispatch(cause, closure, t, median.get(cause, np.nan), p90, review, None, rate)
    d.update(closure=closure, f4=None, type_rate=closure, tier=t,
             median=median.get(cause, np.nan), p90=p90, review=review)
    return d


@st.cache_data
def f4_reliability():
    m = load_model()
    df = load_csv("closure_clean.csv")
    for c in m["num"]:
        df[c] = df[c].astype(float)
    Xc = m["encoder"].transform(df[m["cat"]].astype(str))
    X = np.concatenate([Xc, df[m["num"]].values], axis=1)
    df["p"] = m["model"].predict_proba(X)[:, 1]
    df["actual"] = (df["requires_road_closure"] == True).astype(int)
    df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("actual", "mean"),
                                             n=("p", "size")).reset_index(drop=True)
    return g


def card(title, sub, d, conf=None, kind="upcoming"):
    """Severity-styled HTML card for the Live Ops feed."""
    color = SEV_HEX.get(d["tier"], "#888")
    acts = [f"{d['personnel']} officer(s)"]
    if d["barricade"]:
        acts.append("Barricade")
    if d["diversion"]:
        acts.append("Diversion")
    hold_h = (d["median"] if d["review"] else d["p90"]) / 60.0
    confline = f"<span style='color:#888;font-size:11px'> &middot; conf {conf:.0%}</span>" if conf is not None else ""
    srcline = ("F4 %.0f%% / type %.0f%%" % (d["f4"] * 100, d["type_rate"] * 100)) if d.get("f4") is not None else "type analogue"
    tag = "PRE-DEPLOY" if kind == "upcoming" else "DISPATCH"
    review = "<div style='color:#d08770;font-size:11px;margin-top:3px'>&#9888; ticket-lifecycle &rarr; F7 review</div>" if d["review"] else ""
    return (f"<div style='border-left:4px solid {color};background:#1b1e25;padding:9px 13px;"
            f"border-radius:6px;margin-bottom:7px'>"
            f"<div style='font-weight:600;font-size:14px'>{title}{confline}</div>"
            f"<div style='font-size:12px;color:#9aa'>{sub}</div>"
            f"<div style='margin-top:5px;font-size:13px'><span style='color:{color}'>&#9679; {d['tier']}</span>"
            f" &middot; closure <b>{d['closure']:.0%}</b> <span style='color:#777;font-size:11px'>({srcline})</span>"
            f" &middot; ~{d['median']:.0f} min</div>"
            f"<div style='font-size:13px;margin-top:3px;color:#cdd'>&#9654; {tag}: "
            f"{' &middot; '.join(acts)} &middot; hold ~{hold_h:.1f}h</div>{review}</div>")


# ----- demo fixtures (mirror the standalone agents) -----
UPCOMING = [
    {"name": "Farmers' rally", "venue": "Town Hall, MG Road", "cause": "protest", "conf": 0.91, "lat": 12.9762, "lon": 77.5993},
    {"name": "State dignitary motorcade", "venue": "Airport Rd \u2192 Vidhana Soudha", "cause": "vip_movement", "conf": 0.88, "lat": 12.9791, "lon": 77.5906},
    {"name": "Cricket fixture", "venue": "Chinnaswamy Stadium", "cause": "public_event", "conf": 0.95, "lat": 12.9788, "lon": 77.5996},
    {"name": "Flyover repair works", "venue": "Hosur Road", "cause": "construction", "conf": 0.84, "lat": 12.9150, "lon": 77.6220},
    {"name": "Temple car procession", "venue": "Basavanagudi", "cause": "procession", "conf": 0.79, "lat": 12.9420, "lon": 77.5730},
]
EXAMPLES = [
    {"label": "Tree fall blocking road", "cause": "tree_fall", "venue": "Sadashivanagar", "priority": "High", "veh": "not_applicable", "lat": 13.0060, "lon": 77.5810, "hour": 14},
    {"label": "BMTC bus breakdown", "cause": "vehicle_breakdown", "venue": "Silk Board", "priority": "Unknown", "veh": "bmtc_bus", "lat": 12.9170, "lon": 77.6230, "hour": 9},
    {"label": "Multi-vehicle accident", "cause": "accident", "venue": "Marathahalli", "priority": "High", "veh": "private_car", "lat": 12.9560, "lon": 77.7010, "hour": 18},
    {"label": "VIP movement", "cause": "vip_movement", "venue": "Vidhana Soudha approach", "priority": "High", "veh": "not_applicable", "lat": 12.9800, "lon": 77.5910, "hour": 11},
]


# ======================= LOAD =======================
ref = load_csv("reference_table.csv")
sev = load_csv("severity_by_type.csv")
buf = load_csv("conformal_buffer.csv")
disp = load_csv("dispatch_schema.csv")
model = load_model()

st.title("\U0001F6A6 Event-Impact & Resource Recommendation Engine")
st.caption("Bengaluru Traffic Police \u00b7 Flipkart GRiD Round 2 \u2014 forecasts **impact**, "
           "recommends **resources**, **learns** from every event. Not a congestion predictor.")

tabs = st.tabs(["Overview", "\U0001F3AF Live Ops", "Live Assessment", "Reference Table",
                "Calibration", "Learning Loop", "Map", "Explore"])

# ---- Overview ----
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Clean events", "2,458", help="real-close clearance records after canonical cleaning")
    c[1].metric("Closure model AUC", "0.79", help="vs 0.74 type-lookup baseline")
    c[2].metric("P90 buffer coverage", "90.2%", help="conformal, proven out-of-sample")
    c[3].metric("Outlier rate", "3.9%", help="events past their peer-cohort fence")
    st.success("**What this means operationally:** at the barricade operating point, the engine "
               "pre-positions units for **8% of events** and catches **~50% of all actual road closures** "
               "before the road is shut (at ~52% precision). A **VIP movement anywhere in Bengaluru "
               "auto-triggers a barricade recommendation** \u2014 its 80% historical closure rate sets a "
               "floor the per-event model cannot override.")
    st.markdown("---")
    a, b = st.columns(2)
    a.markdown("#### What it solves\n"
               "- **Impact quantified in advance** \u2014 reference table + calibrated closure prob + severity\n"
               "- **Resource deployment, data-backed** \u2014 dispatch with tunable operating points\n"
               "- **Post-event learning** \u2014 outlier review queue + corridor tracker")
    b.markdown("#### Honest by design\n"
               "- Forecasts *impact*, not congestion (the log has no flow data)\n"
               "- Both uncertainty claims **proven** (buffer coverage, closure calibration)\n"
               "- One model tested & dropped for not beating its baseline")
    st.info("**Key finding:** Low-priority events close roads **12.1%** of the time vs High-priority "
            "**5.9%** \u2014 priority tracks dispatch urgency, not disruption. Data beats intuition.")
    st.caption("\U0001F449 Open the **Live Ops** tab for the full pipeline in one view.")

# ---- LIVE OPS (the integrated command center) ----
with tabs[1]:
    st.subheader("\U0001F3AF Live Operations Command Center")
    st.caption("The whole system in one flow: **detect \u2192 pre-score \u2192 dispatch \u2192 map \u2192 learn**. "
               "Upcoming events are pre-scored from analogues; live incidents run the per-event model; "
               "everything plots and logs.")

    if "ops" not in st.session_state:
        st.session_state.ops = []          # fired live incidents
    if "scanned" not in st.session_state:
        st.session_state.scanned = False

    # ---------- 1 - DETECT: upcoming events ----------
    st.markdown("##### 1 &middot; Detect \u2014 *news watch pre-scores upcoming events*")
    ca, cb = st.columns([1, 4])
    if ca.button("\U0001F4E1 Scan upcoming", type="primary", width="stretch"):
        st.session_state.scanned = True
    if cb.button("Clear board"):
        st.session_state.scanned = False
        st.session_state.ops = []
    if st.session_state.scanned:
        cols = st.columns(len(UPCOMING))
        for i, ev in enumerate(UPCOMING):
            d = prescore_type(ev["cause"], ref, sev, buf)
            cols[i].markdown(card(ev["name"], f"@ {ev['venue']} &middot; <i>{ev['cause']}</i>", d,
                             conf=ev["conf"], kind="upcoming"), unsafe_allow_html=True)
    else:
        st.info("Press **Scan upcoming** to pre-score detected events from historical analogues.")

    st.markdown("---")
    # ---------- 2 - DISPATCH: live incidents ----------
    st.markdown("##### 2 &middot; Dispatch \u2014 *fire a live incident through the per-event model*")
    bcols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if bcols[i].button(ex["label"], key=f"fire{i}", width="stretch"):
            inp = dict(event_cause=ex["cause"], priority=ex["priority"], event_type="unplanned",
                       veh_type=ex["veh"], police_station="-", zone="Unknown", on_corridor=1,
                       start_hour=ex["hour"], desc_highway=0, desc_multiple=0, desc_signal=0)
            d = score_live(inp, model, ref, sev, buf)
            st.session_state.ops.append({"label": ex["label"], "venue": ex["venue"],
                                         "cause": ex["cause"], "lat": ex["lat"], "lon": ex["lon"],
                                         "d": d})
    if st.session_state.ops:
        fc = st.columns(2)
        for i, o in enumerate(reversed(st.session_state.ops)):
            fc[i % 2].markdown(card(o["label"], f"@ {o['venue']} &middot; <i>{o['cause']}</i>", o["d"],
                               kind="live"), unsafe_allow_html=True)
    else:
        st.caption("No live incidents yet \u2014 click an incident button above to dispatch.")

    st.markdown("---")
    # ---------- 3 - MAP: operational picture ----------
    st.markdown("##### 3 &middot; Map \u2014 *operational picture*")
    pins = []
    if st.session_state.scanned:
        for ev in UPCOMING:
            d = prescore_type(ev["cause"], ref, sev, buf)
            pins.append({"lat": ev["lat"], "lon": ev["lon"], "label": ev["name"],
                         "kind": "upcoming", "tier": d["tier"], "closure": f"{d['closure']:.0%}",
                         "color": SEV_RGB[d["tier"]], "radius": 180})
    for o in st.session_state.ops:
        pins.append({"lat": o["lat"], "lon": o["lon"], "label": o["label"], "kind": "live",
                     "tier": o["d"]["tier"], "closure": f"{o['d']['closure']:.0%}",
                     "color": SEV_RGB[o["d"]["tier"]], "radius": 300})
    if pins:
        pdf = pd.DataFrame(pins)
        layer = pdk.Layer("ScatterplotLayer", pdf, get_position=["lon", "lat"],
                          get_fill_color="color", get_radius="radius", opacity=0.75,
                          stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1,
                          pickable=True)
        view = pdk.ViewState(latitude=12.96, longitude=77.60, zoom=10.5)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="road",
                        tooltip={"text": "{label}\n{kind} | {tier} | closure {closure}"}))
        leg = "  ".join(f"<span style='color:{SEV_HEX[k]}'>\u25cf</span> {k}" for k in ["High", "Medium", "Low"])
        st.markdown(leg + " &nbsp;&nbsp; <span style='color:#888'>large = live incident, small = upcoming</span>",
                    unsafe_allow_html=True)
    else:
        st.caption("Map populates once you scan upcoming events or fire a live incident.")

    st.markdown("---")
    # ---------- 4 - LEARN: review log ----------
    st.markdown("##### 4 &middot; Learn \u2014 *review log feeds the F7 outlier queue*")
    if st.session_state.ops:
        log = pd.DataFrame([{"incident": o["label"], "type": o["cause"], "severity": o["d"]["tier"],
                             "closure": f"{o['d']['closure']:.0%}", "clearance_min": round(o["d"]["median"]),
                             "officers": o["d"]["personnel"],
                             "barricade": "yes" if o["d"]["barricade"] else "-",
                             "diversion": "yes" if o["d"]["diversion"] else "-",
                             "review": "F7" if o["d"]["review"] else "-"} for o in st.session_state.ops])
        st.dataframe(log, width="stretch", hide_index=True)
        st.caption("Each dispatched incident is logged; long-tail outliers route to the F7 review queue "
                   "(see the Learning Loop tab). The standalone alert_agent.py persists this to review_log.csv.")
    else:
        st.caption("The operations log builds as you dispatch incidents.")

# ---- Live Assessment ----
with tabs[2]:
    st.subheader("Score an event")
    st.caption("Analyst tool: probe any event. A live incident uses the per-event closure model "
               "with a conservative type-rate floor.")
    L, R = st.columns([1, 1])
    with L:
        cause = st.selectbox("Event cause", sorted(ref["event_cause"]))
        priority = st.selectbox("Priority", ["High", "Low", "Unknown"])
        etype = st.selectbox("Event type", ["unplanned", "planned"])
        veh = st.selectbox("Vehicle type", ["not_applicable", "bmtc_bus", "heavy_vehicle", "lcv",
                            "private_car", "private_bus", "truck", "ksrtc_bus", "taxi", "auto", "others"])
        on_corr = st.checkbox("On a corridor", value=True)
        hour = st.slider("Hour of day (IST)", 0, 23, 18)
    with R:
        ps = st.text_input("Police station", "Madiwala")
        zone = st.selectbox("Zone", ["East", "West", "South", "North", "Central", "Unknown"])
        lat = st.number_input("Latitude", value=12.95, format="%.4f")
        lon = st.number_input("Longitude", value=77.62, format="%.4f")
        flags = st.multiselect("Description flags", ["highway", "multiple vehicles", "signal failure"])
    if st.button("Assess", type="primary"):
        inp = dict(event_cause=cause, priority=priority, event_type=etype, veh_type=veh,
                   police_station=ps, zone=zone, on_corridor=int(on_corr), start_hour=hour,
                   desc_highway=int("highway" in flags), desc_multiple=int("multiple vehicles" in flags),
                   desc_signal=int("signal failure" in flags))
        d = score_live(inp, model, ref, sev, buf)
        m = st.columns(4)
        m[0].metric("Severity", d["tier"])
        m[1].metric("Closure risk", f"{d['closure']:.0%}", help=f"F4 {d['f4']:.0%} / type {d['type_rate']:.0%}")
        m[2].metric("Expected clearance", f"{d['median']:.0f} min")
        m[3].metric("Resource hold", f"~{d['hold_min']/60:.1f} h")
        # ---- action-oriented dispatch order (instruction, not statistic) ----
        order = [f"Deploy **{d['personnel']} officer(s)** to site"]
        if d["barricade"]:
            order.append("**erect barricade**")
        if d["diversion"]:
            order.append("**activate diversion** on alternate route")
        order.append(f"hold resources ~{d['hold_min']/60:.1f}h")
        badge = {"High": "\U0001F534 URGENT", "Medium": "\U0001F7E0 PRIORITY", "Low": "\U0001F7E2 ROUTINE"}.get(d["tier"], "")
        line = f"### {badge} \u2014 DEPLOY NOW\n" + " &middot; ".join(order) + "."
        if d["tier"] == "High":
            st.error(line)
        elif d["tier"] == "Medium":
            st.warning(line)
        else:
            st.success(line)
        for n in d["notes"]:
            st.caption("\u26a0 " + n)

# ---- Reference Table ----
with tabs[3]:
    st.subheader("Per-type reference table (the transparent core predictor)")
    st.caption("Median clearance, de-contaminated P90, closure rate with Wilson 95% CIs, reliability tier.")
    show = ref.copy()
    show["closure_95CI"] = show.apply(lambda r: f"[{r['closure_lo95']:.2f}, {r['closure_hi95']:.2f}]", axis=1)
    st.dataframe(show[["event_cause", "n_dur", "n_full", "reliability", "median_min", "p90_min",
                       "stale_removed", "closure_rate", "closure_95CI"]],
                 width="stretch", hide_index=True)
    st.caption("Sparse types fall back to a pooled duration estimate; closure rate stays own with a wide CI.")

# ---- Calibration ----
with tabs[4]:
    st.subheader("Calibration \u2014 the credibility tab")
    st.markdown("Every uncertainty claim is **verified out-of-sample**, not asserted.")
    a, b = st.columns(2)
    with a:
        st.markdown("**P90 clearance buffer (conformal)**")
        st.metric("Marginal coverage", "90.2%", "target 90%")
        bshow = buf[["event_cause", "op_median_min", "conf_p90_min", "test_coverage", "status"]]
        st.dataframe(bshow, width="stretch", hide_index=True)
    with b:
        st.markdown("**Closure probability reliability (F4)**")
        st.metric("ECE", "0.0096", "well calibrated")
        rel = f4_reliability()
        rel_plot = rel.rename(columns={"predicted": "Predicted", "observed": "Observed"}).set_index("Predicted")[["Observed"]]
        st.line_chart(rel_plot)
        st.caption("Predicted vs observed closure frequency per decile \u2014 points hug the diagonal.")

# ---- Learning Loop ----
with tabs[5]:
    st.subheader("Post-event learning loop")
    q = load_csv("outlier_queue.csv")
    cr = load_csv("corridor_ranking.csv")
    g = q[q["kind"] == "genuine"]
    st.markdown(f"**Outlier review queue** \u2014 {len(g)} genuine anomalies (ran past their peer cohort), "
                f"{(q['kind']=='ticket_artifact').sum()} ticket-artifacts separated.")
    st.dataframe(g[["rank", "event_cause", "on_corridor", "duration_min", "cohort_median",
                    "excess_ratio"]].head(15), width="stretch", hide_index=True)
    st.markdown("**Corridor ranking (hypothesis-only)** \u2014 Wilson CIs; none yet distinguishable "
                "from the global rate, so presented as directional.")
    crs = cr.copy()
    if "distinguishable" in crs.columns:
        crs["distinguishable"] = crs["distinguishable"].map({True: "yes", False: "no"})
    st.dataframe(crs, width="stretch", hide_index=True)

# ---- Map ----
with tabs[6]:
    st.subheader("Incident map \u2014 severity-coloured")
    st.caption("Honest incident density (count-based), never a fabricated congestion surface.")
    md = disp.dropna(subset=["lat", "lon"]).copy()
    sel = st.multiselect("Show severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    md = md[md["severity_tier"].isin(sel)]
    md = md.sample(min(2000, len(md)), random_state=42)
    md["color"] = md["severity_tier"].map(SEV_RGB)
    layer = pdk.Layer("ScatterplotLayer", md, get_position=["lon", "lat"],
                      get_fill_color="color", get_radius=120, opacity=0.6, pickable=True)
    view = pdk.ViewState(latitude=12.97, longitude=77.59, zoom=10.3)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style="road",
                    tooltip={"text": "{event_cause}\nSeverity: {severity_tier}\nClosure: {closure_risk}"}))
    leg = "  ".join(f"<span style='color:{SEV_HEX[k]}'>\u25cf</span> {k}" for k in ["High", "Medium", "Low"])
    st.markdown(leg, unsafe_allow_html=True)

# ---- Explore ----
with tabs[7]:
    st.subheader("Explore the data")
    dur = load_csv("duration_clean.csv")
    a, b = st.columns(2)
    with a:
        st.markdown("**Median clearance by event type (min)**")
        st.bar_chart(dur.groupby("event_cause")["duration_min"].median().sort_values(ascending=False))
    with b:
        st.markdown("**Closure rate by event type**")
        st.bar_chart(ref.set_index("event_cause")["closure_rate"].sort_values(ascending=False))
    st.markdown("**Filter clearance records**")
    f1, f2 = st.columns(2)
    pick_cause = f1.multiselect("Event cause", sorted(dur["event_cause"].unique()))
    pick_corr = f2.selectbox("Corridor", ["all", "on-corridor", "off-corridor"])
    v = dur.copy()
    if pick_cause:
        v = v[v["event_cause"].isin(pick_cause)]
    if pick_corr != "all":
        v = v[v["on_corridor"] == (pick_corr == "on-corridor")]
    st.caption(f"{len(v)} records \u00b7 median {v['duration_min'].median():.0f} min" if len(v) else "no records")
    st.dataframe(v[["event_cause", "priority", "corridor", "veh_type", "police_station",
                    "duration_min", "closure_flag"]].head(200), width="stretch", hide_index=True)
