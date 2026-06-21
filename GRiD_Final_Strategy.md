# Event-Impact & Resource Recommendation Engine
## Final Strategy — Flipkart GRiD Round 2

> **The thesis in one sentence.** Most teams will try to predict congestion from an incident log and fail visibly. We forecast *impact*, recommend *resources*, and *learn* from every event — three claims, each traceable to the raw CSV, each mapping directly to a stated gap. Every data-quality decision in this document was **verified against the actual data**, not assumed.

---

## 1. What We're Actually Solving

The brief states three operational gaps and asks: *how can historical and real-time data forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?*

| Gap | What it means operationally | How we solve it |
|-----|----------------------------|-----------------|
| Event impact not quantified in advance | Officers decide deployment by gut feel | Severity score + clearance forecast + closure probability per event |
| Resource deployment is experience-driven | No data-backed manpower/barricade plan | Rules scaffold whose thresholds are **tied to calibrated closure probability as explicit operating points** — not arbitrary constants |
| No post-event learning system | Same mistakes repeated | Outlier review queue + corridor pattern tracker |

**The reframe that keeps every claim defensible.** The dataset is an *incident log*, not a *flow* dataset — it has no speed, volume, or occupancy. So "impact" cannot mean congestion. We operationalise impact as **clearance duration + closure probability + a transparent severity composite**. This is the distinction that lets us deliver the full brief — *impact* **and** *resources* — without inventing a single number.

**On the resource half (historically the weaker half of any submission for this problem).** There is no manpower-deployed ground truth in the data, so nothing about resourcing can be *learned*. Instead of hiding that behind a black box, we make the resource layer defensible by anchoring every dispatch threshold to the closure model's calibrated probability and stating the precision/recall trade-off at each operating point (Section 4, F6). "Deploy a barricade at risk ≥ 0.30" becomes "at 0.30 you pre-position on *X%* of events and catch *Y%* of true closures" — an operating point a judge can interrogate, not a magic constant.

---

## 2. Data Reality — Verified Against the CSV

**Dataset:** ASTraM incident log — **8,173 events × 46 columns**, Bengaluru, Nov 2023–Apr 2024.

Every decision below was checked against the raw file. These are the load-bearing facts:

### 2.1 The timestamps are local IST, mislabeled UTC — hour-of-day is recoverable

The raw values carry a `+00` (UTC) suffix, but read **as stored** they peak at **21:00** with a textbook traffic shape: an evening peak (19:00–23:00), an early-morning cluster (04:00–07:00), and a midday dead zone (10:00–16:00). A naive `+00 → IST (+5:30)` conversion slides that real 21:00 peak to 02:30 and produces a spurious "2 AM peak."

**Verification:** `created_date` (report time) shows the *identical* 21:00 peak, and `created − start` is a median of **1.5 minutes** — the two clocks agree with each other. The values are already Bengaluru local time.

**Consequence:** hour-of-day is **not corrupted**. We recover it — but we scope its use deliberately (Section 7.3): it is **not a model feature** (it adds nothing once you condition on event type, see 2.5) and is used only for (a) correctness, and (b) the shift-availability layer in the resource recommendation, which is a real operational lever because fewer officers are on duty at night.

> We do **not** claim a causal story for the night-heavy pattern. We tested whether it reflects Bengaluru's daytime heavy-vehicle restriction and the data does **not** support it — the supposedly-banned types (`heavy_vehicle` 0.43, `truck` 0.46) are the *least* night-concentrated, while `private_car` (0.54) and `bmtc_bus` (0.54), which face no curfew, are the *most*. So we state only the descriptive fact (incidents are evening/overnight-heavy) and make no causal claim. *This restraint is the moat — see Section 11.*

### 2.2 The rescue-rejection predicate (exact, reproducible)

The duration models train only on events with a **real** close time. The exclusion rule is one line:

> **Drop rows where `status == "closed"` but `closed_datetime` is null** — these ~**3,956** "rescue" rows only receive a fabricated duration via `modified_datetime`.

Rows that *have* a genuine `closed_datetime` are kept regardless of who last modified them. (The earlier "reject everything the bot `FKUSR00001` touched" framing was wrong: that bot is `last_modified_by_id` on **7,091** rows, including 2,098 of the clean rows — rejecting on it would gut the dataset to 362 rows.)

After the predicate, applying a positivity filter and a 24-hour cap leaves **≈2,460 clean-duration rows** (≈2,456 after removing 3 test/demo rows). All duration models train on this subset; the closure model uses the full population (2.3).

### 2.3 The closure flag is set at creation — full population, no selection bias

`requires_road_closure` is populated for **all 8,173** rows (overall closure rate **8.3%**, 676 events). It is set at event creation, so it survives later modification and is trustworthy even on bot-modified rows. This is why the closure model trains on the full 8,173 with **no selection bias** — state this explicitly, because bot-modified rows have a *lower* closure rate (0.076 vs 0.128) and a judge could otherwise read that as a leak.

### 2.4 Columns we drop, and why

`end_datetime` is **94.2% missing** (only 475 populated; 3 are future-dated, max 2027) — unusable; we compute duration from `closed_datetime` instead. Also dropped: `comment`, `map_file`, `meta_data`, `age_of_truck`, `direction`, the sentinel `endlat/long`, and the `resolved_*` family. Case inconsistencies normalised (`Debris`/`debris` → `debris`), 3 `test_demo` rows removed, categorical nulls filled `"Unknown"`.

### 2.5 The accuracy ceiling is real

`event_cause` alone explains R² ≈ 0.22 of duration variance — the other ~78% is unrecorded ground conditions, irreducible noise. We add richer features (2.6) and **measure the full-feature R² at build**; we do not anchor a slide on 0.22. If the richer set lifts it to ~0.30, we say so; if it doesn't, that is itself an honesty point. The right accuracy story is **calibrated uncertainty**, not lowest error (Section 7).

### 2.6 What genuinely adds signal (all verified)

- **Sub-type splitting inside `vehicle_breakdown`** (the dominant 4,896-event category) — the single highest-value feature. `veh_type` is ~**100% populated within breakdowns** (only 9 nulls), so the split has full coverage where it matters, and it's a real gradient, not just two points: buses and heavy vehicles clear slower (`bmtc_bus` 47.8 min, `truck` 44.6, `heavy_vehicle` 42.8, `private_bus` 42.6) than light vehicles (`private_car` 29.5, `auto` 27.6, `ksrtc_bus` 32.3, `taxi` 33.2). Splitting the largest category along this axis is where most of the real accuracy lift lives.
- **`description` text flags** (~6,813 non-null, 16.6% missing): boolean/TF-IDF keywords ("highway", "multiple vehicles", "signal failure") differentiate complexity within a type. Missing → zeroed flags.
- **Spatial units — `police_station` is primary, `zone`/`junction` are supplementary.** Only `police_station` is fully populated (all 8,173). `zone` is **57.9% null** and `junction` is **69.3% null**, so filling them "Unknown" makes them mostly-Unknown and weak as model features — use `police_station` as the spatial feature, treat `zone`/`junction` as supplements *where present* (and for display), not co-equal predictors. Raw lat/long is clean (0 nulls, all inside the Bengaluru bbox) and kept for the map.

> **Note on the `veh_type` 40% overall null:** it is *structural*, not dirt — `veh_type` is N/A for non-vehicle events (tree falls, potholes, water logging). It is populated exactly where it's used. We fill non-vehicle rows `not_applicable`, not `Unknown`.

### 2.7 Step 0 — The Canonical Cleaning Pass (run and audited *first*)

Cleaning is a standalone first step, not logic buried inside the model. One runnable pass produces the canonical datasets everything downstream reads, plus a printed audit log so every decision is reproducible.

**Ordered pipeline:**

1. Load; assert shape **8,173 × 46**; assert **0 duplicate IDs** (verified: 0 exact-dup rows, all IDs unique).
2. Drop the **3 `test_demo`** rows; normalise categorical case/whitespace (collapses the one real collision, `Debris`→`debris`).
3. Parse datetimes **without timezone conversion** (already local IST — see 2.1); compute `duration = closed_datetime − start_datetime`.
4. **Closure population** = all surviving rows (flag = `requires_road_closure`, set at creation → trustworthy on the full 8,173).
5. **Duration population** = drop `status=="closed"` with null `closed_datetime` (~3,956 rescue rows), keep real-close rows, apply `0 < dur ≤ 1440` (drops 3 negatives; flags the 666 >24h rows `near_24h_cap`). → **2,458 rows** (canonical count).
6. Apply the explicit **missing-data policy** (below).
7. Emit **`closure_clean.csv`** + **`duration_clean.csv`** + a **printed audit log** (before/after row counts at each step).

**Missing-data policy (per used column):**

| Column | Null % | Policy |
|--------|-------:|--------|
| `event_cause`, `requires_road_closure`, `status`, `event_type`, `police_station`, lat/long | 0% | use as-is (lat/long verified in-bbox) |
| `closed_datetime` | 61.6% | drives the rescue predicate; duration only where present |
| `veh_type` | 40.2% (structural) | `not_applicable` for non-vehicle events; ~100% present within breakdowns |
| `description` | 16.6% | text flags zeroed when missing |
| `corridor` | 0.2% | on/off-corridor binary (`Non-corridor`→off); null→off |
| `priority` | <0.1% (2 rows) | fill `Unknown` |
| `zone` | 57.9% | supplementary feature, used where present |
| `junction` | 69.3% | display/supplementary only — **not** a model feature |

**Drop entirely (dead/unused):** `end_datetime` (94.2% null, 3 future-dated rows), `comment`, `map_file`, `meta_data`, `age_of_truck`, `direction`, `endlatitude/endlongitude`, the `resolved_*` family, `veh_no`, `kgid`, `gba_identifier`, `route_path`, `cargo_material`, `reason_breakdown`, `authenticated`, `modified_datetime`, and the ID columns (kept only long enough to apply the rescue check, then dropped).

---

## 3. System Architecture (Scope-Disciplined)

We build a **bulletproof core + one labeled extension**. Self-hosted routing is cut (see Section 9).

```
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                           │
│        [ASTraM CSV — cleaned per Section 2]              │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                CORE PREDICTION ENGINE                    │
│  F1 Data Prep   F2 Reference Table ★   F3 Duration       │
│  F4 Closure Risk (calibrated)   F5 Severity Index        │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                DECISION & RESPONSE LAYER                 │
│  F6 Dispatch Recommendation (operating-point thresholds) │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│              POST-EVENT LEARNING LAYER (F7)              │
│   Outlier Review Queue      Corridor Pattern Tracker     │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│        STREAMLIT DASHBOARD  +  News Watch (extension)    │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Core Prediction Engine

### F1 — Data Preparation
F1 is the executable form of the **Step 0 canonical cleaning pass (2.7)** — it runs *first*, emits `closure_clean.csv` + `duration_clean.csv` + the printed audit log, and is verified before any model touches the data. It produces the two populations from Section 2: **duration population** (≈2,460 rows, the rescue predicate + 24h cap) and **closure population** (8,173 rows), and prints the honesty report: rescue-rejection count, selection skew (the duration subsample over-represents breakdowns), and the seasonal blind spot (no monsoon window — `month` ≠ season; weather has near-zero importance).

### F2 — Per-Type Reference Table ★ (Primary Predictor)
The centrepiece: one row per event cause, fully readable by an operator, and competitive with the ML model on overall error. **Verified numbers:**

| Event Cause | Median Dur | Closure Rate | n (dur) | Reliability |
|-------------|-----------:|-------------:|--------:|-------------|
| Construction | 296 min | 26.5% | 51 | reliable |
| Road Conditions | 246 min | 12.4% | 31 | reliable |
| Water Logging | 113 min | 8.5% | 100 | reliable |
| Tree Fall | 91 min | 39.4% | 103 | reliable |
| Others | 75 min | 8.6% | 234 | reliable |
| Congestion | 72 min | 4.4% | 22 | directional |
| Vehicle Breakdown | 41 min | 4.3% | 1,783 | reliable |
| Accident | 40 min | 3.0% | 87 | reliable |
| Procession | 37 min | 26.4% | 13 | directional |
| Pot Holes | 36 min | 2.4% | 32 | reliable |
| Public Event | pooled | 46.4% | 0 | sparse (dur) |
| VIP Movement | pooled | 80.0% | 20 | sparse (dur) |

**Reliability tiers gate usage:** `reliable` (n ≥ 30) uses own stats; `directional` (10–29) uses own but flags uncertainty; `sparse` (n < 10) falls back to the pooled global estimate. VIP/public have high closure but no clean duration — duration falls back to pooled, **clearly flagged**.

**P90 buffer + de-contamination:** compute each type's P90 after removing its own IQR outliers (Q3 + 1.5×IQR); keep a visible `stale_removed` count as a data-quality signal. **Long-tail honesty:** the 24h cap censors genuine multi-day events (666 closed rows exceed 24h), so long-type medians/P90s (Construction, Road Conditions) are reported as **lower bounds**, and the `near_24h_cap` flag separates left-open tickets from genuine long events.

### F3 — Duration Refinement Model (Secondary)
`HistGradientBoostingRegressor` on log-duration with **direct quantile loss** at α = 0.10/0.50/0.90 (P10/P50/P90 in one pass — better calibrated than a post-hoc buffer). Subordinate to F2: only overrides the table *downward* for short types.

**Features:** one-hot `event_cause`, `corridor`, `priority`, `event_type`; the `veh_type × event_cause` sub-type split (2.6); `description` text flags; spatial units `zone`/`junction`/`police_station`; raw lat/long.
**Excluded:** hour-of-day (adds nothing within type — 2.5/7.3), weather (near-zero importance), anything post-resolution.
**Honesty baked in:** every fit prints baseline comparison (constant vs type-lookup vs model). They tie overall; the model wins on short events and loses on the long tail — report both.

### F4 — Closure Risk Model (Calibrated Probability)
"Does this event need barricades or diversions?" Trained on all 8,173 (no selection bias, 2.3).
- **Class weighting OFF** — weighting inflates probabilities 2–5× and destroys calibration. Output is a **calibrated probability**, never a binary flag.
- Features beyond F2: `zone`, `junction`, `police_station`, `event_type`, `veh_type`.
- **Expected lift is small** (prototype AUC ≈ 0.80 vs type-lookup ≈ 0.74 — confirm at build), so the **type-lookup rate stays primary** and the model is a thin spatial refinement. Hour-of-day is excluded: closure varies by time at the raw level (≈0.064 night vs ≈0.099 day) but is **flat within event type** (vehicle_breakdown 0.038/0.048, construction 0.266/0.264) — the raw gap is pure composition.

> **Headline finding for the judges (verified):** Low-priority events have a **12.1%** closure rate vs High-priority at **5.9%**. Priority tracks *dispatch urgency*, not *disruption level*. This single inversion demonstrates the value of data over intuition — **lead the deck with it.**

### F5 — Severity Index (Transparent Composite)
A single interpretable impact ranking, openly constructed (not learned):
```
severity = 0.5 × normalize(expected_duration) + 0.5 × closure_risk
```
`expected_duration` uses the robust F2 type-median; normalisation clipped at 5th/95th pct; tiers Low/Med/High at frozen quantile cutpoints (<60th / 60–85th / ≥85th); **bounds and cutpoints frozen at fit time and persisted** so one live event scores on the same basis as the batch.
**Caveat to state:** duration and closure are correlated, so the two components are not fully independent — report `corr(duration, closure)` and frame severity as a *weighted decision scaffold*, not two orthogonal signals. Weights are visible and tunable.

### F6 — Dispatch Recommendation (Operating-Point Rules)
Converts engine outputs into actions. No manpower ground truth exists, so every constant lives in a marked `CONFIG` block for BTP to calibrate — **but the closure thresholds are presented as operating points, not magic numbers.**

- **Personnel:** Low → 1 officer, Medium → 2, High → 3; +1 if closure risk ≥ 0.5. *Shift modifier:* night-shift availability adjustment (the one place recovered time-of-day pays off).
- **Barricade:** if closure risk ≥ 0.30 → *report the trade-off:* "at 0.30, barricades pre-positioned on N% of events, catching M% of true closures."
- **Diversion:** if closure risk ≥ 0.50 or severity High.
- **Resource commit time:** P90 duration — plan for the tail, not the mean.

**Honest seam:** "nearest available unit" assignment is intentionally not built — it needs live fleet positions the log lacks. This is the documented connector to live ops data.

**Example output**
```
Event: Tree Fall — Bellary Road
Severity: HIGH | Closure Risk: 39% | Expected Clearance: 91 min (P90 lower-bounded ~3h)
→ Deploy 3 officers (+night-shift modifier if applicable)
→ Barricade recommended (risk 39% ≥ 0.30; at this threshold ~X% pre-positioned, ~Y% of closures caught)
→ No diversion (risk < 0.50, severity threshold not met)
→ Hold resources up to ~3h
```

---

## 5. The One Extension — News & Event Watch Agent (Minimal Spine)

**Purpose:** detect upcoming Bengaluru events so the engine can pre-estimate impact from historical analogues — directly addressing "quantified in advance" and the brief's "real-time data."

**Scoped hard for Round 2:** one reliable source → entity extraction → analogue match → pre-score. Not six sources, not a heavy agent framework.
```
One source (BTP advisory / single RSS) → extract {name, location, time, type, scale}
   → map event_type to nearest event_cause in F2
   → pre-score: expected clearance, closure risk, severity tier
   → push to dashboard "upcoming events" panel + dispatch recommendation
```
**LLM-assisted extraction (Claude API, `claude-sonnet-4-6`)** for unstructured notices, returning a structured JSON record.

**Labeled honestly as a forward-looking extension, not a co-equal pillar:** VIP + public events are ~2% of the data, so the analogue lookup is the best achievable pre-estimate — we don't claim more precision than the reference table supports.

---

## 6. Post-Event Learning Loop (F7)

Solves the third gap. Two products of deliberately different evidential strength:

**6.1 Outlier Review Queue (defensible).** Flags events past their peer cohort's (event type × on/off-corridor) IQR upper fence (Q3 + 1.5×IQR, min cohort 8) — ~3.9% genuine outliers. Genuine outliers ranked by excess vs cohort median; `near_24h_cap` artifacts sort to the bottom. **Never auto-diagnoses blame** — flags for human review only.

**6.2 Corridor Ranking (hypothesis-only).** Volume-normalised outlier rate per corridor with **Wilson confidence intervals**. On current data, no corridor is statistically distinguishable from the global ~3.9% rate — presented explicitly as directional, sharpening as data accumulates.

**Why this wins:** no other team addresses this gap. Every event logged becomes institutional memory.

---

## 7. Accuracy & Calibration Strategy

**The honest ceiling:** ~78% of duration variance is irreducible. The right story is **best-calibrated uncertainty so officers plan correctly**, not lowest MedAE.

**What genuinely helps:** sub-type splitting, text flags, spatial units, direct quantile regression (all Section 2.6 / F3). **What not to chase:** model complexity — the bottleneck is missing features, not expressiveness.

### 7.1 The calibration deliverable — *this proves the moat, build it first*
The one claim neither analysis pass has yet validated against data. Before anything else after F2/F3:
- **Duration coverage test (out-of-sample):** does ~90% of held-out actuals fall under the predicted P90? Plot empirical coverage vs nominal for P10/P50/P90.
- **Closure reliability diagram:** bin predicted probabilities, plot vs observed frequency; report Brier score and calibration-in-the-large.

This converts "best-calibrated uncertainty" from an assertion into a **plot a judge can see**. It is the cheapest, highest-leverage credibility win in the project.

### 7.2 Accuracy claim for judges
"We extracted every real signal in this log — sub-type features, text keywords, spatial granularity — produced direct quantile bands, and **verified their coverage out-of-sample**. The residual uncertainty is documented, bounded, and honest."

### 7.3 Time-of-day, scoped
Recovered (timezone fix, 2.1) for **correctness and the night-shift resource modifier only**. It is **not** a duration or closure feature — within-type effects are ~3 min / ~0 respectively — and carries **no causal narrative** (2.1). Sold as rigor, not as an accuracy booster.

---

## 8. Dashboard (Streamlit)

- **Landing** — hero cards: closure AUC (confirmed at build) · 12+ event types profiled · ~3.9% outlier rate · ≈2,460 clean events. One-line scope statement: *"Impact forecasting and resource recommendation from incident data — not a congestion predictor."* Upcoming-events panel (news agent).
- **Live Assessment** — input (cause, corridor, priority, vehicle, lat/long); hint *"Event type drives most of the result"*; outputs: expected clearance (median + P90), closure risk %, severity tier, dispatch recommendation, reliability warning for sparse types.
- **Reference Table** — F2 on screen with n, median, closure rate, Wilson CI, `stale_removed`. The transparency centrepiece.
- **Calibration** — the coverage plot + reliability diagram from 7.1. *Few teams will show this; it is the credibility tab.*
- **Learning Loop** — outlier queue (genuine vs artifact split) + corridor ranking with Wilson CIs and the directional-only caveat.
- **Explore / Map** — duration-by-type boxplot, closure-rate-by-type bar, incident-density heatmap (grey basemap, red ramp, ~50% opacity, **count-based — never a fake congestion surface**). News-agent events overlaid as pins. No hour/day cuts presented as causal.

---

## 9. Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Core ML | sklearn `HistGradientBoostingRegressor` | No install friction; quantile loss |
| LLM (news extraction) | Claude API (`claude-sonnet-4-6`) | Structured extraction from unstructured notices |
| Dashboard | Streamlit + pydeck | Fast; density-map support |
| Persistence | Pickle (model bundle) + CSV (reference table) + JSON (severity/calibration params) | Simple, reproducible |
| Event source (extension) | One reliable feed (BTP advisory / RSS) | Free, demoable |

**Cut for Round 2:** self-hosted OSRM routing (highest-effort, highest-risk single item; "closure-aware routing around one segment" adds least to the core thesis) and the Telegram bot (nice-to-have, not load-bearing). Both are documented as future extensions, not built.

---

## 10. What We Don't Build (and Why)

| Excluded | Reason |
|---------|--------|
| Traffic flow / congestion prediction | No speed, volume, or occupancy in the log |
| Hour-of-day as a model feature | Adds nothing once conditioned on event type (verified) |
| A causal "commercial-vehicle ban" narrative | Data falsifies it — banned types are *least* night-concentrated |
| CCTV / computer vision; adaptive signals | No camera feeds, no flow data, no signal authority |
| LSTM/GRU/sequence models | Events are irregular; no regular time series |
| Weather as a load-bearing feature | Near-zero importance; no monsoon window |
| Self-hosted routing (this round) | Highest infra risk; least core value |
| Any fabricated metric ("47% less congestion") | No ground truth — every number is traceable to the CSV |

---

## 11. Honesty Ledger — The Moat, Made Literal

Each row is a decision we can defend with the data. *This table is itself a slide.*

| Claim under scrutiny | What we checked | Decision |
|----------------------|-----------------|----------|
| "Clock is corrupted, hour-of-day is dead" | Raw peak 21:00; `created−start` median 1.5 min; created_date agrees | **Reversed** — timestamps are local IST; hour recovered, scoped to shift layer |
| "Overnight cluster = daytime truck ban" | Night-share by veh_type: heavy 0.43 / truck 0.46 *lowest*, car 0.54 *highest* | **Rejected** — make no causal claim, only the descriptive fact |
| "Reject everything the bot touched" | Bot is `last_modified` on 7,091 rows incl. 2,098 clean rows | **Replaced** with exact predicate: drop `closed` status with null close time (~3,956) |
| "Closure model has no selection bias" | Bot rows closure 0.076 vs 0.128 | **Justified** — `requires_road_closure` is set at creation; state it explicitly |
| "Best-calibrated uncertainty" | Not yet tested | **Build the coverage test first** (7.1) — prove it, don't assert it |
| "R² ≈ 0.22 is the ceiling" | Cause-only figure; richer features unmeasured | **Measure full-feature R² at build**; report honestly either way |
| "zone/junction/police_station are co-equal spatial features" | Null %: zone 57.9%, junction 69.3%, police_station 0% | **Corrected** — `police_station` is the primary spatial unit; `zone`/`junction` are supplementary-where-present (2.6) |
| "veh_type is 40% dirty/missing" | Within `vehicle_breakdown` it's ~100% populated (9 nulls) | **Reclassified** — structural N/A for non-vehicle events, not dirt; sub-type split has full coverage |

---

## 12. The Differentiator in One Sentence

We predict *impact*, recommend *resources* against calibrated operating points, and *learn* from every event — three claims, each backed by traceable, **verified** data, each mapping directly to a stated gap. The teams that redefine the problem to look impressive lose in Q&A; the team that shows exactly what the data can and cannot support wins it.

---

## Appendix — Build Order (when you greenlight)

1. **Step 0 — canonical cleaning (F1), run and verified *first*** — apply the exact rescue predicate (2.2), the per-column missing-data policy with `not_applicable`/`Unknown` fills (2.7), the case fix and column drops; emit `closure_clean.csv` + `duration_clean.csv` + the **printed audit log** (before/after counts at each step). **We review the audit log together and lock the canonical datasets before any modeling runs.**
2. **F2 reference table** — reproduce the verified numbers; reliability tiers; P90 de-contamination; long-tail lower-bound flags.
3. **7.1 calibration harness** — coverage test + reliability diagram scaffolding *before* tuning anything. If the bands don't cover, fix that before adding features.
4. **F3 / F4 / F5** — quantile duration model, calibrated closure model, frozen severity composite; measure full-feature R²; confirm AUC lift.
5. **F6** — operating-point dispatch with the CONFIG block and night-shift modifier.
6. **Dashboard** — Reference Table + Calibration tabs first (the credibility core), then Live Assessment, Learning Loop, Map.
7. **F7 learning loop** — outlier queue + corridor ranking with Wilson CIs.
8. **News agent spine** — single source → extract → analogue → pre-score; labeled as an extension.

*Validate the calibration coverage (step 3) before building on any timing claim — it is the one remaining load-bearing assertion in the whole strategy.*
