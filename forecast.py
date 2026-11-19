"""
TERRANEX - Landslide alert engine (live 24h alerts + historical replay).

Combines:
  * Rainfall and soil moisture from Open-Meteo (free, no API key)
      - live mode   : last 24h / last 7 days up to now
      - replay mode : the same windows ending at the end of a chosen PAST date
  * Historical landslide inventory (susceptibility of each district)
  * Reference values learned from the inventory (rainfall / soil moisture recorded
    on past landslide dates)

It is a transparent, explainable scoring model - NOT a trained ML model and NOT an
official warning. See METHOD_NOTE below.
"""
import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from flask import jsonify, request

try:
    from imerg import imerg_last_24h
except Exception as _exc:          # imerg.py missing/broken -> app still works with Open-Meteo
    print("IMERG module not loaded:", _exc)
    imerg_last_24h = None

LIVE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_SECONDS = 30 * 60
BATCH_SIZE = 50
REPLAY_MIN_DATE = "1950-01-01"
REPLAY_CACHE_LIMIT = 30

WEIGHTS = {"rain24": 0.35, "rain7": 0.25, "soil": 0.20, "history": 0.20}
HIGH_AT = 70
ELEVATED_AT = 45

_cache = {"time": 0.0, "payload": None}
_replay_cache = {}

METHOD_NOTE = (
    "Each district gets two scores: NOW (last 24 h, observed) and NEXT 24 H (predicted from the "
    "weather-model forecast). The headline level is the worse of the two. "
    "Score (0-100) = weighted mix of 24h rainfall (35%), 7-day rainfall (25%), "
    "soil moisture (20%) and historical landslide density of the district (20%). Rainfall "
    "and soil moisture are compared with values recorded on past landslide dates in the "
    "GSI inventory. A district cannot be flagged High/Elevated unless recent rainfall is "
    "meaningfully high. This is a decision-support indicator, not an official warning or "
    "a probability. Always follow IMD / NDMA / state disaster-management alerts."
)

FORECAST_NOTE = (
    "Next-24h rainfall and soil moisture are NOT measured values: they come from the Open-Meteo "
    "numerical weather-prediction forecast at each district's centre point. Mountain rainfall is "
    "very local, so actual amounts can differ a lot. The next-24h score applies the same "
    "explainable scoring to that forecast; it is not a trained ML model or a probability."
)

REPLAY_NOTE = (
    "Replay is an in-sample sanity check: the scoring thresholds come from the same GSI "
    "inventory (records of the replayed date are excluded from them). It is not a formal "
    "accuracy test; a proper train/test validation needs a trained model."
)

MEASURES = {
    "High": [
        "Issue a local alert; be ready to move families living on or below steep slopes, near road cuts and streams.",
        "Avoid travel on hill roads and known slide-prone stretches, especially at night and in heavy rain.",
        "Keep district disaster teams (SDRF/NDRF) on standby and keep relief routes and shelters ready.",
        "Watch for warning signs: new cracks, tilted trees/poles, muddy spring water, sudden change in stream flow.",
        "Cross-check with the latest IMD warning and your district disaster management authority.",
    ],
    "Elevated": [
        "Increase monitoring of vulnerable slopes, road cuts and settlements below hillsides.",
        "Avoid unnecessary travel on hill roads during heavy rain; keep emergency contacts handy.",
        "Pre-position JCBs / clearing crews on key highways; prepare community warning (SMS / loudspeaker).",
        "Check IMD forecast for the next 24-48 hours.",
    ],
    "Low": [
        "Routine monitoring is enough. Keep drainage clear and stay updated on weather forecasts.",
    ],
}

EXTRA_MEASURES = {
    "rain": "Clear drains, culverts and side ditches so water does not pond on slopes.",
    "soil": "Stop excavation, slope cutting and heavy loading on already saturated slopes.",
    "history": "Inspect previously failed slopes and old scarps; check retaining walls and slope protection.",
}


# ------------------------------------------------------------------ helpers
def _clip01(x):
    return float(min(max(x, 0.0), 1.0))


def _quantile(series, q, default):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) >= 20 else default


def reference_values(df):
    """Rainfall / soil moisture typical of past landslide events in the inventory."""
    ref = {
        "rain24": max(_quantile(df["rainfall_24h"], 0.75, 60.0), 1.0),
        "rain7": max(_quantile(df["rainfall_7d"], 0.75, 150.0), 1.0),
        "soil_low": _quantile(df["soil_moisture"], 0.25, 0.25),
        "soil_high": _quantile(df["soil_moisture"], 0.75, 0.40),
    }
    if ref["soil_high"] <= ref["soil_low"]:
        ref["soil_high"] = ref["soil_low"] + 0.05
    return ref


def district_table(df, df_history=None):
    """One row per district. Coordinates come from df; the history count comes from
    df_history (defaults to df) so replay can exclude the replayed date."""
    hist = df if df_history is None else df_history

    base = (
        df[df["District"] != "Not specified"]
        .groupby(["State", "District"])
        .agg(latitude=("Latitude", "median"), longitude=("Longitude", "median"))
        .reset_index()
    )
    counts = (
        hist[hist["District"] != "Not specified"]
        .groupby(["State", "District"])
        .size()
        .rename("records")
        .reset_index()
    )
    g = base.merge(counts, on=["State", "District"], how="left")
    g["records"] = g["records"].fillna(0).astype(int)
    max_records = max(int(g["records"].max()), 1)
    g["history_factor"] = np.log1p(g["records"]) / np.log1p(max_records)
    return g


def fetch_weather(points, url, extra_params):
    """points: list of (lat, lon). Returns one Open-Meteo result per point."""
    results = []
    for i in range(0, len(points), BATCH_SIZE):
        chunk = points[i:i + BATCH_SIZE]
        params = {
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in chunk),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in chunk),
            "hourly": "precipitation,soil_moisture_0_to_7cm",
            "timezone": "UTC",
        }
        params.update(extra_params)
        req = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "TERRANEX/1.0"})
        with urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        results.extend(data if isinstance(data, list) else [data])
    return results


def extract_observation(entry, now, with_future=False):
    hourly = entry.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    stamps = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]
    past = [i for i, t in enumerate(stamps) if t <= now]
    if not past:
        return None
    end = past[-1] + 1

    rain = pd.to_numeric(pd.Series(hourly.get("precipitation") or []), errors="coerce")
    soil = pd.to_numeric(pd.Series(hourly.get("soil_moisture_0_to_7cm") or []), errors="coerce")

    r24 = rain.iloc[max(0, end - 24):end]
    r7 = rain.iloc[max(0, end - 168):end]
    if r24.notna().sum() < 12:          # too many gaps -> do not guess
        return None
    s24 = soil.iloc[max(0, end - 24):end].dropna()
    obs = {
        "rain24": float(r24.sum()),
        "rain7": float(r7.sum()),
        "soil": float(s24.mean()) if len(s24) else None,
    }

    if with_future:
        fut = rain.iloc[end:end + 24]
        if fut.notna().sum() >= 12:
            fsoil = soil.iloc[end:end + 24].dropna()
            stop = end + len(fut)
            peak_i = int(np.nanargmax(fut.values))
            obs["next"] = {
                "rain24": float(fut.sum()),
                "rain7": float(rain.iloc[max(0, stop - 168):stop].sum()),
                "soil": float(fsoil.mean()) if len(fsoil) else None,
                "hourly": [round(float(x), 2) if pd.notna(x) else 0.0 for x in fut],
                "peak_mm": float(fut.iloc[peak_i]),
                "peak_time": times[end + peak_i][:16] + ":00Z",
            }
    return obs


def score_district(obs, history_factor, ref):
    f = {
        "rain24": _clip01(obs["rain24"] / ref["rain24"]),
        "rain7": _clip01(obs["rain7"] / ref["rain7"]),
        "history": _clip01(history_factor),
    }
    if obs["soil"] is not None:
        f["soil"] = _clip01((obs["soil"] - ref["soil_low"]) / (ref["soil_high"] - ref["soil_low"]))

    total_w = sum(WEIGHTS[k] for k in f)
    score = 100 * sum(WEIGHTS[k] * f[k] for k in f) / total_w

    # No rainfall trigger -> never raise an alert on history alone
    if max(f["rain24"], f["rain7"]) < 0.35:
        score = min(score, ELEVATED_AT - 1)

    level = "High" if score >= HIGH_AT else "Elevated" if score >= ELEVATED_AT else "Low"
    return round(score, 1), level, f


def build_reasons(obs, f, records, ref):
    src = obs.get("rain_source", "Open-Meteo model")
    reasons = [
        f"Rainfall in the last 24 hours [{src}]: {obs['rain24']:.1f} mm "
        f"({f['rain24'] * 100:.0f}% of the level typical on past landslide dates, {ref['rain24']:.0f} mm).",
        f"Rainfall in the last 7 days: {obs['rain7']:.1f} mm "
        f"({f['rain7'] * 100:.0f}% of the typical past-landslide level, {ref['rain7']:.0f} mm).",
    ]
    if obs["soil"] is not None:
        sm = f.get("soil", 0)
        state = "close to saturated / wet" if sm >= 0.6 else "moderate" if sm >= 0.3 else "relatively dry"
        reasons.append(f"Soil moisture (0-7 cm, 24h mean): {obs['soil']:.3f} m3/m3 - {state}.")
    else:
        reasons.append("Soil moisture data not available for this location; score uses the other factors.")
    reasons.append(f"{records} historical landslide records in this district (past slope failures).")
    return reasons


def _ist(iso_utc):
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")) + timedelta(hours=5, minutes=30)
    return dt.strftime("%H:%M")


def build_forecast_reasons(now_obs, nxt, n_f, ref):
    out = [
        f"PREDICTED rainfall in the next 24 hours: {nxt['rain24']:.1f} mm "
        f"({n_f['rain24'] * 100:.0f}% of the typical past-landslide level, {ref['rain24']:.0f} mm); "
        f"last 24 hours were {now_obs['rain24']:.1f} mm.",
        f"Heaviest predicted hour: {nxt['peak_mm']:.1f} mm/h at {_ist(nxt['peak_time'])} IST.",
        f"7-day total including the next 24 h would reach {nxt['rain7']:.1f} mm.",
    ]
    if nxt["soil"] is not None:
        out.append(f"Predicted soil moisture (0-7 cm, next-24h mean): {nxt['soil']:.3f} m3/m3.")
    return out


def build_measures(level, f):
    items = list(MEASURES[level])
    if level != "Low":
        if max(f["rain24"], f["rain7"]) >= 0.6:
            items.append(EXTRA_MEASURES["rain"])
        if f.get("soil", 0) >= 0.6:
            items.append(EXTRA_MEASURES["soil"])
        if f["history"] >= 0.6:
            items.append(EXTRA_MEASURES["history"])
    return items


def _validate_replay(df, as_of, rows):
    """Compare AI levels with landslides actually recorded on the replayed date."""
    events = df[df["landslide_date"] == as_of]
    by_district = events.groupby(["State", "District"]).size()
    level_by = {(r["state"], r["district"]): r for r in rows}

    checks = []
    for (state, district), n in by_district.items():
        r = level_by.get((state, district))
        checks.append({
            "state": state,
            "district": district,
            "landslides": int(n),
            "level": r["level"] if r else "No data",
            "score": r["score"] if r else None,
        })
    checks.sort(key=lambda c: c["landslides"], reverse=True)

    with_data = [c for c in checks if c["level"] != "No data"]
    flagged = [c for c in with_data if c["level"] in ("High", "Elevated")]
    all_flagged = sum(1 for r in rows if r["level"] in ("High", "Elevated"))

    actual_events = [
        {"latitude": float(r.Latitude), "longitude": float(r.Longitude),
         "state": r.State, "district": r.District}
        for r in events.head(300).itertuples()
    ]
    return {
        "date": as_of,
        "recorded_landslides": int(len(events)),
        "event_districts": len(checks),
        "event_districts_with_data": len(with_data),
        "event_districts_flagged": len(flagged),
        "districts_flagged_overall": all_flagged,
        "districts_assessed": len(rows),
        "checks": checks[:25],
        "actual_events": actual_events,
        "note": REPLAY_NOTE,
    }


def compute_forecast(df, as_of=None):
    if as_of:
        history = df[df["landslide_date"] != as_of]
        ref = reference_values(history)
        districts = district_table(df, history)
        day = datetime.strptime(as_of, "%Y-%m-%d")
        now = day.replace(hour=23, tzinfo=timezone.utc)
        start = (day - timedelta(days=7)).strftime("%Y-%m-%d")
        points = list(zip(districts["latitude"], districts["longitude"]))
        weather = fetch_weather(points, ARCHIVE_URL, {"start_date": start, "end_date": as_of})
    else:
        ref = reference_values(df)
        districts = district_table(df)
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        points = list(zip(districts["latitude"], districts["longitude"]))
        weather = fetch_weather(points, LIVE_URL, {"past_days": 7, "forecast_days": 2})

    # Observed last-24h rainfall from NASA IMERG satellite data (live mode only; falls back safely)
    imerg_vals = [None] * len(points)
    imerg_meta = {"status": "off", "message": "Observed rainfall: Open-Meteo model data (replay uses archived data)."}
    if not as_of and imerg_last_24h is not None:
        imerg_vals, imerg_meta = imerg_last_24h(points)

    rows = []
    for i, ((_, d), entry) in enumerate(zip(districts.iterrows(), weather)):
        obs = extract_observation(entry, now, with_future=not as_of)
        if obs is None:
            continue
        model_rain24 = obs["rain24"]
        obs["rain_source"] = "Open-Meteo model" if not as_of else "ERA5 archive"
        if imerg_vals[i] is not None:
            obs["rain24"] = float(imerg_vals[i])
            obs["rain_source"] = "NASA IMERG satellite, observed"
        score, level, f = score_district(obs, d["history_factor"], ref)
        records = int(d["records"])
        reasons = build_reasons(obs, f, records, ref)

        nxt = obs.get("next")
        extra = {}
        eff_score, eff_level, eff_f, driver = score, level, f, "last_24h"
        if nxt:
            n_score, n_level, n_f = score_district(nxt, d["history_factor"], ref)
            if n_score > score:
                eff_score, eff_level, eff_f, driver = n_score, n_level, n_f, "next_24h"
            diff = n_score - score
            extra = {
                "now_score": score, "now_level": level,
                "next_score": n_score, "next_level": n_level,
                "trend": "Rising" if diff >= 5 else "Easing" if diff <= -5 else "Steady",
                "driver": driver,
                "next_rain_24h_mm": round(nxt["rain24"], 1),
                "next_soil_moisture": None if nxt["soil"] is None else round(nxt["soil"], 3),
                "next_peak_mm_per_hr": round(nxt["peak_mm"], 1),
                "next_peak_time": nxt["peak_time"],
                "next_hourly_mm": nxt["hourly"],
            }
            reasons = build_forecast_reasons(obs, nxt, n_f, ref) + reasons
        rows.append({
            "state": d["State"],
            "district": d["District"],
            "latitude": float(d["latitude"]),
            "longitude": float(d["longitude"]),
            "score": eff_score,
            "level": eff_level,
            "rain_24h_mm": round(obs["rain24"], 1),
            "rain_24h_source": obs["rain_source"],
            "rain_24h_model_mm": round(model_rain24, 1),
            "rain_7d_mm": round(obs["rain7"], 1),
            "soil_moisture": None if obs["soil"] is None else round(obs["soil"], 3),
            "historical_records": records,
            "reasons": reasons,
            "measures": build_measures(eff_level, eff_f),
            **extra,
        })

    if not rows:
        raise RuntimeError("No usable weather data was returned.")

    rows.sort(key=lambda r: r["score"], reverse=True)
    payload = {
        "mode": "replay" if as_of else "live",
        "as_of": as_of,
        "generated_at": now.isoformat(),
        "source": ("Open-Meteo archive (ERA5 reanalysis)" if as_of else "Open-Meteo forecast API")
                  + " - hourly precipitation, soil moisture 0-7 cm",
        "method": METHOD_NOTE,
        "districts_assessed": len(rows),
        "districts_without_data": int(len(districts) - len(rows)),
        "summary": {lvl: sum(1 for r in rows if r["level"] == lvl) for lvl in ("High", "Elevated", "Low")},
        "districts": rows,
    }
    if not as_of:
        payload["imerg"] = imerg_meta
        payload["forecast_note"] = FORECAST_NOTE
        payload["summary_now"] = {l: sum(1 for r in rows if r.get("now_level") == l) for l in ("High", "Elevated", "Low")}
        payload["summary_next"] = {l: sum(1 for r in rows if r.get("next_level") == l) for l in ("High", "Elevated", "Low")}
    if as_of:
        payload["validation"] = _validate_replay(df, as_of, rows)
    return payload


# ------------------------------------------------------------------ Flask hooks
def register_forecast(app, df):
    @app.route("/api/forecast")
    def forecast():
        if _cache["payload"] and time.time() - _cache["time"] < CACHE_SECONDS:
            return jsonify(_cache["payload"])
        try:
            payload = compute_forecast(df)
        except Exception as exc:
            print("Forecast error:", exc)
            return jsonify({
                "error": "Live weather data could not be fetched right now. "
                         "No alert is shown rather than showing a guess. Please try again later."
            }), 503
        _cache.update(time=time.time(), payload=payload)
        return jsonify(payload)

    @app.route("/api/replay-dates")
    def replay_dates():
        """Dates with the most recorded landslides - good candidates for a replay."""
        counts = df["landslide_date"].dropna().value_counts()
        latest_ok = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        out = [
            {"date": d, "landslides": int(n)}
            for d, n in counts.items()
            if REPLAY_MIN_DATE <= d <= latest_ok
        ][:12]
        return jsonify(out)

    @app.route("/api/replay")
    def replay():
        as_of = (request.args.get("date") or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of):
            return jsonify({"error": "Please give a date as YYYY-MM-DD."}), 400
        try:
            datetime.strptime(as_of, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "That is not a valid date."}), 400

        latest_ok = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        if as_of < REPLAY_MIN_DATE or as_of > latest_ok:
            return jsonify({
                "error": f"Replay supports dates from {REPLAY_MIN_DATE} to {latest_ok} "
                         "(archive data lags by a few days)."
            }), 400

        if as_of in _replay_cache:
            return jsonify(_replay_cache[as_of])
        try:
            payload = compute_forecast(df, as_of=as_of)
        except Exception as exc:
            print("Replay error:", exc)
            return jsonify({
                "error": "Archived weather data could not be fetched right now "
                         "(possibly a free-tier rate limit). Please try again in a minute."
            }), 503

        if len(_replay_cache) >= REPLAY_CACHE_LIMIT:
            _replay_cache.pop(next(iter(_replay_cache)))
        _replay_cache[as_of] = payload
        return jsonify(payload)