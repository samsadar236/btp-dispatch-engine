"""
Mappls Visualization Layer  -  thin, swappable renderer over the F6 dispatch schema
Flipkart GRiD Round 2 | Event-Impact & Resource Recommendation Engine

The engine emits MAP-AGNOSTIC dispatch records.  This layer consumes them and
nothing more - the core never imports a map.  It provides:

  1. build_geojson()       dispatch records -> standard GeoJSON (severity colour + popup)
  2. reverse_geocode()     Mappls Reverse Geocode interface -> fills `location_name`
  3. render_mappls_html()  Mappls Web Maps SDK v3.0 (vector) render            [needs API key]
  4. render_preview_html() Leaflet + OpenStreetMap render of the SAME GeoJSON  [no key, runs now]

Swap Mappls <-> Leaflet by changing only the renderer; the GeoJSON contract is identical.
"""
import pandas as pd
import numpy as np
import json

import os
_BASE = os.path.dirname(os.path.abspath(__file__))  # this folder (map/) - where outputs are written
_DATA = os.environ.get("BTP_DATA_DIR", os.path.join(os.path.dirname(_BASE), "data"))  # ../data - where the schema is read from
SCHEMA = os.path.join(_DATA, "dispatch_schema.csv")
GEOJSON = os.path.join(_BASE, "dispatch_markers.geojson")
MAPPLS_HTML = os.path.join(_BASE, "dispatch_map_mappls.html")
PREVIEW_HTML = os.path.join(_BASE, "dispatch_map_preview.html")

SEVERITY_COLOR = {"High": "#d62728", "Medium": "#ff8c00", "Low": "#2ca02c"}
BLR_CENTER = [12.972, 77.594]


def popup_html(r):
    acts = [f"{int(r['recommended_personnel'])} officer(s)"]
    if r["barricade"]:
        acts.append("Barricade")
    if r["diversion"]:
        acts.append("Diversion")
    loc = r["location_name"] if isinstance(r.get("location_name"), str) and r["location_name"] else f"{r['lat']:.4f}, {r['lon']:.4f}"
    review = "<div style='color:#b00'>! ticket-lifecycle - route to review</div>" if r["review_flag"] else ""
    return (f"<div style='font:13px sans-serif;min-width:200px'>"
            f"<b>{str(r['event_cause']).replace('_',' ').title()}</b>"
            f" &nbsp;<span style='color:{SEVERITY_COLOR.get(r['severity_tier'],'#555')}'>"
            f"&#9679; {r['severity_tier']}</span><br>"
            f"<small>{loc}</small><hr style='margin:4px 0'>"
            f"Closure risk: <b>{r['closure_risk']:.0%}</b><br>"
            f"Clearance: {int(r['expected_clearance_min'])} min &nbsp;|&nbsp; hold ~{r['resource_hold_min']/60:.1f}h<br>"
            f"<b>Deploy:</b> {' &bull; '.join(acts)}{review}</div>")


def build_geojson(df, max_per_class=300):
    """Stratified sample (all severities represented) -> GeoJSON FeatureCollection."""
    parts = []
    for tier in ["High", "Medium", "Low"]:
        sub = df[df["severity_tier"] == tier]
        parts.append(sub.sample(min(max_per_class, len(sub)), random_state=42) if len(sub) else sub)
    sample = pd.concat(parts)
    feats = []
    for _, r in sample.iterrows():
        if pd.isna(r["lat"]) or pd.isna(r["lon"]):
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
            "properties": {"severity": r["severity_tier"],
                           "color": SEVERITY_COLOR.get(r["severity_tier"], "#555"),
                           "popupHtml": popup_html(r)},
        })
    return {"type": "FeatureCollection", "features": feats}


def reverse_geocode(lat, lon, token="<MAPPLS_TOKEN>"):
    """[INTERFACE] Fill `location_name` from coordinates at deploy.
    GET https://apis.mappls.com/advancedmaps/v1/{token}/rev_geocode?lat={lat}&lng={lon}
    -> formatted_address + nearby landmark.  No network here; returns None."""
    return None


def render_mappls_html(geojson, api_key="<MAPPLS_API_KEY>"):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="initial-scale=1.0">
<title>BTP Dispatch Map - Mappls</title>
<style>html,body,#map{{margin:0;padding:0;width:100%;height:100vh}}
.legend{{position:absolute;bottom:18px;left:12px;background:#fff;padding:8px 12px;border-radius:6px;
font:13px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3);z-index:999}}</style>
<!-- Mappls Web Maps SDK v3.0 (vector). Insert your key/token below. -->
<script src="https://apis.mappls.com/advancedmaps/api/{api_key}/map_sdk?layer=vector&v=3.0&callback=loadMap" defer async></script>
</head><body>
<div id="map"></div>
<div class="legend"><b>Severity</b><br>
<span style="color:#d62728">&#9679;</span> High &nbsp;
<span style="color:#ff8c00">&#9679;</span> Medium &nbsp;
<span style="color:#2ca02c">&#9679;</span> Low</div>
<script>
const DISPATCH = {json.dumps(geojson)};
function loadMap() {{
  const map = new mappls.Map('map', {{center: {BLR_CENTER}, zoom: 11}});
  map.on('load', function() {{
    DISPATCH.features.forEach(function(f) {{
      const c = f.geometry.coordinates;            // [lng, lat]
      // HTML marker keeps severity colour; popup carries the dispatch recommendation.
      const el = document.createElement('div');
      el.style.cssText = 'width:12px;height:12px;border-radius:50%;border:1px solid #fff;background:'+f.properties.color;
      new mappls.Marker({{map:map, position:{{lat:c[1], lng:c[0]}}, html:el,
        popupHtml:f.properties.popupHtml}});
    }});
  }});
}}
</script></body></html>"""


def render_preview_html(geojson):
    """No-key Leaflet + OSM preview of the identical GeoJSON (verify rendering now)."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="initial-scale=1.0">
<title>BTP Dispatch Map - Preview (no key)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{margin:0;padding:0;width:100%;height:100vh}}
.legend{{background:#fff;padding:8px 12px;border-radius:6px;font:13px sans-serif;line-height:1.5}}</style>
</head><body>
<div id="map"></div>
<script>
const DISPATCH = {json.dumps(geojson)};
const map = L.map('map').setView({BLR_CENTER}, 11);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{attribution:'&copy; OpenStreetMap', maxZoom:19}}).addTo(map);
L.geoJSON(DISPATCH, {{
  pointToLayer: (f, latlng) => L.circleMarker(latlng,
    {{radius:6, color:'#fff', weight:1, fillColor:f.properties.color, fillOpacity:0.85}}),
  onEachFeature: (f, layer) => layer.bindPopup(f.properties.popupHtml)
}}).addTo(map);
const lg = L.control({{position:'bottomleft'}});
lg.onAdd = () => {{ const d = L.DomUtil.create('div','legend');
  d.innerHTML = '<b>Severity</b><br><span style="color:#d62728">&#9679;</span> High '+
  '<span style="color:#ff8c00">&#9679;</span> Medium <span style="color:#2ca02c">&#9679;</span> Low'; return d; }};
lg.addTo(map);
</script></body></html>"""


def main():
    df = pd.read_csv(SCHEMA, low_memory=False)
    gj = build_geojson(df)
    json.dump(gj, open(GEOJSON, "w"))
    open(MAPPLS_HTML, "w").write(render_mappls_html(gj))
    open(PREVIEW_HTML, "w").write(render_preview_html(gj))

    # ---- Verify the adapter output here (the render itself happens in the browser) ----
    by_sev = {}
    for f in gj["features"]:
        by_sev[f["properties"]["severity"]] = by_sev.get(f["properties"]["severity"], 0) + 1
    print("=" * 72)
    print(" MAPPLS VIZ LAYER  -  adapter output verified")
    print("=" * 72)
    print(f"  dispatch records in : {len(df)}")
    print(f"  GeoJSON features out : {len(gj['features'])}  (stratified sample)")
    print(f"  by severity colour   : " + "  ".join(
        f"{k} {by_sev.get(k,0)} ({SEVERITY_COLOR[k]})" for k in ['High','Medium','Low']))
    coords = [f["geometry"]["coordinates"] for f in gj["features"]]
    lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
    print(f"  bbox (lon/lat)       : [{min(lons):.3f},{min(lats):.3f}] -> [{max(lons):.3f},{max(lats):.3f}]  (Bengaluru)")
    print(f"  sample popup         : {gj['features'][0]['properties']['popupHtml'][:90]}...")
    print("-" * 72)
    print(f"  saved GeoJSON        -> {GEOJSON}")
    print(f"  Mappls (production)  -> {MAPPLS_HTML}   (insert your Mappls key)")
    print(f"  Leaflet preview      -> {PREVIEW_HTML}   (open in a browser now, no key)")
    print("  location_name is filled by reverse_geocode() at deploy; engine stays map-free.")


if __name__ == "__main__":
    main()
