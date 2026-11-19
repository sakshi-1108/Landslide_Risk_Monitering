"""
TERRANEX - NASA GPM IMERG (Early Run) observed rainfall for the last 24 hours.

IMERG is satellite-based rainfall *observation* (not a weather-model forecast):
half-hourly, ~10 km (0.1 deg) grid. The Early Run appears ~4 hours after the
observation time, so the window returned here ends roughly 4+ hours before "now".
The exact window is returned in `meta` so the UI can show it honestly.

Setup (once):
  1. Free account: https://urs.earthdata.nasa.gov/
  2. In your Earthdata profile -> "Applications" -> Authorized Apps, approve
     "NASA GESDISC DATA ARCHIVE".
  3. pip install earthaccess xarray h5netcdf
  4. Set credentials as environment variables (never put them in the code):
        Windows (PowerShell):  $env:EARTHDATA_USERNAME="you"; $env:EARTHDATA_PASSWORD="secret"
        Linux / macOS:         export EARTHDATA_USERNAME=you EARTHDATA_PASSWORD=secret
     (or have a ~/.netrc / _netrc file with the urs.earthdata.nasa.gov entry)

Quick standalone test (prints the last-24h rainfall at one point):
        python imerg.py 27.3 92.4

This module NEVER raises into the app: if anything fails (no login, no network,
library missing) it returns None values and a plain-language message, and the
app falls back to Open-Meteo for that run.
"""
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

SHORT_NAME = "GPM_3IMERGHHE"      # IMERG Early Run, half-hourly. Version is chosen automatically.
STEPS = 48                        # 48 half-hours = 24 h
MIN_STEPS = 40                    # need >= 40 of 48 valid steps, otherwise no value (do not guess)
SEARCH_HOURS = 32                 # look back far enough to still find 48 steps despite ~4 h latency
BOX_PAD = 0.3                     # degrees of padding around the district points
CACHE_SECONDS = 60 * 60

_cache = {"time": 0.0, "key": None, "value": None}
_lock = threading.Lock()
_state = {"thread": None, "error": None, "error_time": 0.0, "started": 0.0}
RETRY_AFTER_ERROR = 10 * 60


def _meta(status, message, **extra):
    out = {"status": status, "message": message, "run": "IMERG Early (satellite, ~10 km)"}
    out.update(extra)
    return out


# ------------------------------------------------------------------ granule helpers
def _begin(g):
    return g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]


def _end(g):
    return g["umm"]["TemporalExtent"]["RangeDateTime"]["EndingDateTime"]


def _version(g):
    try:
        return str(g["umm"]["CollectionReference"]["Version"])
    except Exception:
        return "0"


def pick_latest_granules(results, steps=STEPS):
    """One granule per half-hour (newest dataset version wins), newest `steps` of them, oldest first."""
    best = {}
    for g in results:
        b = _begin(g)
        if b not in best or _version(g) > _version(best[b]):
            best[b] = g
    keys = sorted(best)[-steps:]
    return [best[k] for k in keys]


# ------------------------------------------------------------------ reading
def _login(earthaccess):
    if os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD"):
        auth = earthaccess.login(strategy="environment")
    else:
        auth = earthaccess.login(strategy="netrc")
    if not auth or not getattr(auth, "authenticated", True):
        raise RuntimeError("Earthdata login failed")


def _read_points(file_objs, lats, lons):
    """Sum of rainfall (mm) per point over the given half-hourly files + number of valid steps."""
    import xarray as xr

    n = len(lats)
    total = np.zeros(n)
    valid = np.zeros(n, dtype=int)

    w, e = min(lons) - BOX_PAD, max(lons) + BOX_PAD
    s, nn = min(lats) - BOX_PAD, max(lats) + BOX_PAD
    la = xr.DataArray(np.asarray(lats, dtype=float), dims="pt")
    lo = xr.DataArray(np.asarray(lons, dtype=float), dims="pt")

    t0 = time.time()
    for k, f in enumerate(file_objs, 1):
        if k == 1 or k % 6 == 0 or k == len(file_objs):
            print(f"IMERG: reading file {k}/{len(file_objs)} ({time.time() - t0:.0f}s elapsed)", flush=True)
        try:
            ds = xr.open_dataset(f, group="Grid", engine="h5netcdf", decode_times=False)
            da = ds["precipitation"].isel(time=0)
            sub = da.sel(lon=slice(w, e), lat=slice(s, nn))
            if sub.size == 0:                       # in case latitude is stored descending
                sub = da.sel(lon=slice(w, e), lat=slice(nn, s))
            sub = sub.load()
            vals = sub.sel(lon=lo, lat=la, method="nearest").values.astype(float)
            ds.close()
        except Exception as exc:                    # one bad file must not kill the whole run
            print("IMERG: could not read a granule:", exc)
            continue
        vals = np.where(np.isfinite(vals) & (vals >= 0), vals, np.nan)   # fill values -> NaN
        ok = ~np.isnan(vals)
        total[ok] += vals[ok] * 0.5                 # mm/hr over half an hour -> mm
        valid[ok] += 1
    return total, valid


# ------------------------------------------------------------------ public
def imerg_last_24h(points):
    """points: list of (lat, lon). Returns (values, meta).
    values[i] = observed rainfall in mm over the latest 24 h available (None if unavailable)."""
    n = len(points)
    none = [None] * n
    if n == 0:
        return none, _meta("unavailable", "No points requested.")

    key = tuple((round(a, 3), round(b, 3)) for a, b in points)
    if _cache["value"] is not None and _cache["key"] == key and time.time() - _cache["time"] < CACHE_SECONDS:
        return _cache["value"]

    try:
        import earthaccess
    except ImportError:
        return none, _meta("unavailable",
                           "NASA IMERG not used: install the libraries (pip install earthaccess xarray h5netcdf). "
                           "Using Open-Meteo model data instead.")

    if not (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")):
        netrc_ok = any(os.path.exists(os.path.expanduser(p)) for p in ("~/.netrc", "~/_netrc"))
        if not netrc_ok:
            return none, _meta("unavailable",
                               "NASA IMERG not used: no Earthdata credentials found "
                               "(set EARTHDATA_USERNAME / EARTHDATA_PASSWORD). Using Open-Meteo model data instead.")

    try:
        _login(earthaccess)
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=SEARCH_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
        stop = now.strftime("%Y-%m-%dT%H:%M:%S")
        results = earthaccess.search_data(short_name=SHORT_NAME, temporal=(start, stop), count=200)
        granules = pick_latest_granules(results)
        if len(granules) < MIN_STEPS:
            raise RuntimeError(f"only {len(granules)} IMERG half-hour files found (need {MIN_STEPS}+)")

        print(f"IMERG: found {len(granules)} half-hourly files, opening them (can take a few minutes the first time)...", flush=True)
        files = earthaccess.open(granules)
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        total, valid = _read_points(files, lats, lons)

        values = []
        for t, v in zip(total, valid):
            if v < MIN_STEPS:
                values.append(None)
            else:
                values.append(float(t * STEPS / v))     # small gaps scaled up (never under-count rain)
        if all(v is None for v in values):
            raise RuntimeError("no valid IMERG values could be read")

        w_start, w_end = _begin(granules[0]), _end(granules[-1])
        meta = _meta("used",
                     f"Observed rainfall from NASA IMERG satellite data, window {w_start[:16]}Z to {w_end[:16]}Z "
                     f"(latest available; the Early Run lags real time by about 4 hours).",
                     window_start=w_start, window_end=w_end, steps=len(granules))
        out = (values, meta)
        _cache.update(time=time.time(), key=key, value=out)
        return out
    except Exception as exc:
        print("IMERG error:", exc)
        return none, _meta("unavailable", f"NASA IMERG could not be used ({exc}). Using Open-Meteo model data instead.")


def _worker(points):
    values, meta = imerg_last_24h(points)            # blocking call, fills _cache when successful
    if meta.get("status") != "used":
        _state["error"] = meta.get("message")
        _state["error_time"] = time.time()
    else:
        _state["error"] = None


def imerg_last_24h_async(points):
    """Non-blocking version used by the web app.
    Returns immediately: cached IMERG values if available, otherwise (None values, meta) and
    starts loading in a background thread. Click 'Run Live Analysis' again once it has finished."""
    n = len(points)
    none = [None] * n
    key = tuple((round(a, 3), round(b, 3)) for a, b in points)

    cached_ok = _cache["value"] is not None and _cache["key"] == key
    fresh = cached_ok and time.time() - _cache["time"] < CACHE_SECONDS

    with _lock:
        running = _state["thread"] is not None and _state["thread"].is_alive()
        recent_error = _state["error"] and time.time() - _state["error_time"] < RETRY_AFTER_ERROR
        if not fresh and not running and not recent_error:
            th = threading.Thread(target=_worker, args=(points,), daemon=True)
            _state.update(thread=th, started=time.time())
            th.start()
            running = True

    if cached_ok:                      # fresh, or stale but better than nothing while it refreshes
        return _cache["value"]
    if running:
        waited = int(time.time() - _state["started"])
        return none, _meta("loading",
                           f"NASA IMERG is loading in the background ({waited}s so far; the first load can take "
                           "several minutes). Showing Open-Meteo model data for now - click Run Live Analysis "
                           "again in a few minutes to use the NASA observed data.")
    return none, _meta("unavailable", _state["error"] or "NASA IMERG is not available right now. Using Open-Meteo model data.")


def _open_meteo_same_window(lat, lon, window_start, window_end):
    """Open-Meteo model rainfall (mm) over exactly the same window, for an apples-to-apples check."""
    import json
    from urllib.request import Request, urlopen

    url = ("https://api.open-meteo.com/v1/forecast?latitude=%.4f&longitude=%.4f"
           "&hourly=precipitation&past_days=3&forecast_days=1&timezone=UTC" % (lat, lon))
    with urlopen(Request(url, headers={"User-Agent": "TERRANEX/1.0"}), timeout=60) as r:
        data = json.load(r)
    s = datetime.fromisoformat(window_start.replace("Z", "+00:00"))
    e = datetime.fromisoformat(window_end.replace("Z", "+00:00")) + timedelta(seconds=1)
    e = e.replace(minute=0, second=0, microsecond=0) + (timedelta(hours=1) if e.minute or e.second else timedelta(0))
    total, hours = 0.0, 0
    for stamp, mm in zip(data["hourly"]["time"], data["hourly"]["precipitation"]):
        ts = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
        if s < ts <= e and mm is not None:      # Open-Meteo value at hour T = rain in the hour before T
            total += mm
            hours += 1
    return total, hours


if __name__ == "__main__":
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 27.3
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else 92.4
    vals, meta = imerg_last_24h([(lat, lon)])
    print(meta["message"])
    print(f"NASA IMERG (satellite, observed)  at ({lat}, {lon}): {vals[0]} mm")
    if meta.get("window_start"):
        try:
            mm, hrs = _open_meteo_same_window(lat, lon, meta["window_start"], meta["window_end"])
            print(f"Open-Meteo (model) for the SAME window       : {mm:.1f} mm  ({hrs} hourly values)")
        except Exception as exc:
            print("Could not fetch the Open-Meteo comparison:", exc)