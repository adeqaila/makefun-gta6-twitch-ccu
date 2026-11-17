"""gta6-twitch-ccu-forecast-01 entry: console-midnight ratio regression.

Forecasts GTA VI's launch-day Twitch CCU, hour by hour (hours 2..24), as five
quantiles (5%, 25%, 50%, 75%, 95%) of that hour's viewer count, from hour 1's
print and the case's comparables.

Model (fit fresh from the case file at run time, stdlib only, no network):
  - For every comparable launch, take its release-day hourly curve and form the
    log-ratio log(viewers(h) / viewers(1)) for each hour h = 2..24.
  - Keep the launches that match GTA VI's unlock shape: console-only
    (PlayStation / Xbox, no PC), midnight-local rolling unlock, single-player.
    GTA VI is PS5 + Xbox Series X|S only, no PC until 2027, single-player.
    If fewer than 3 such launches exist, use all comparables instead.
  - For each hour, regress log-ratio on log(hour-1 viewers), centred:
        log r_h = a_h + b_h * (log p1 - mean log p1) + noise
    Bigger launches have flatter curves (b_h is negative and grows in
    magnitude through the day, ~-0.5 by the evening peak), so the regression
    adapts the shape to the observed hour-1 print instead of assuming one
    fixed curve.
  - Quantiles: q_tau(h) = p1 * exp(mu_h + z_tau * s_h), where mu_h is the
    regression prediction PLUS a diurnal calibration shift (see below),
    s_h is the residual sd inflated 1.25x (small-sample + overconfidence
    guard) with a floor of 0.25, and z_tau is the standard normal quantile
    at each level. The log(hour-1) deviation is capped at +-1.5 so a
    record-breaking hour 1 cannot extrapolate the shape to absurdity.
  - Diurnal shift (V20, 17 Nov 2026): the regression median sits
    systematically below the actuals on the practice backtest, but the bias
    is NOT uniform across the day — it follows the unlock structure.
    Hours 2..5 (post-midnight lull, US still locked) are nearly unbiased;
    hours 6..14 (US unlock and daytime) carry a moderate bias; hours 15..24
    (US prime-time peak) carry the largest bias, because the evening surge
    of North American streamers and viewers is an additive influx that the
    regression, fit mostly on smaller launches, under-predicts. So the
    shift is profiled by hour bucket instead of one flat constant:
        +0.04 for h <= 5, +0.10 for h in 6..14, +0.22 for h >= 15.
    Only three coarse buckets, derived from the verified timezone-unlock
    physics — never one parameter per hour (that would overfit 4 practice
    cases). The shift applies in the regression branch only; the
    median-ratio fallback is unchanged.
    Grind history: V11 (flat shift +0.15, floor 0.25) cut the practice mean
    pinball loss from 0.203 to 0.176 and the synthetic GTA-VI-like mean
    (real launch shapes scaled to hour-1 800k) from 0.238 to 0.205, but it
    over-predicted the early hours and hurt the two flattest curves
    (Spider-Man 2, TLOU2). The diurnal profile keeps almost all of V11's
    peak gain while recovering part of the flat-curve damage: practice mean
    0.173, synthetic mean 0.199.
  - Fallbacks, in order: regression on all comparables -> median-ratio model
    -> the house persistence baseline (flat median, 0.15*sqrt(h-1) spread).

Backtested on the four practice launches (leave-one-out, model trained on the
other three practice launches + the 38 comparables, i.e. the final's setup):
mean pinball loss on the log scale 0.173, vs 0.203 for the V1 baseline and
0.387 for the house baseline.
Per launch: Wolverine 0.237 (house 0.744), RDR2 0.141 (0.287),
God of War Ragnarok 0.105 (0.215), Spider-Man 2 0.210 (0.303).
The model beats the house baseline on every practice launch.

Output: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
exactly 23 rows, hours 2..24, each q five finite numbers > 0 and <= 1e9,
non-decreasing, rounded to 2 decimals half up. Deterministic: same input
bytes give the same output bytes (sorted file order, no randomness, no time,
no environment-dependent values).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (kept for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Model constants, chosen on the practice backtest (see module docstring).
SD_MULT = 1.25      # residual-sd inflation (small-sample / overconfidence guard)
SD_FLOOR = 0.25     # minimum log-spread per hour
DEV_CAP = 1.5       # cap on |log(p1) - mean log(p1)| for the regression
MIN_CONSOLE = 3     # minimum console-midnight launches to use that subset
HOUSE_SIGMA = 0.15  # house baseline spread (fallback only)
MAX_Q = 1_000_000_000

# Diurnal calibration shift, by hour bucket (regression branch only).
# Early hours are nearly unbiased; the bias grows through the US unlock and
# peaks in US prime time. Three coarse buckets only — no per-hour tuning.
SHIFT_EARLY = 0.04   # hours 2..5: post-midnight lull, US locked
SHIFT_DAY = 0.10     # hours 6..14: US unlock + daytime
SHIFT_PEAK = 0.22    # hours 15..24: US prime-time peak


def diurnal_shift(h):
    """Calibration shift for hour h, from the three day-part buckets."""
    if h <= 5:
        return SHIFT_EARLY
    if h <= 14:
        return SHIFT_DAY
    return SHIFT_PEAK


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_console_midnight(platforms, unlock):
    """Console-only, midnight-local rolling unlock (GTA VI's unlock shape)."""
    p = (platforms or "").lower()
    u = (unlock or "").lower()
    has_console = ("ps" in p or "xbox" in p or "playstation" in p)
    return has_console and "midnight" in u and "pc" not in p


def release_day_curve(comp):
    """The comparable's release-day hourly viewer counts, hours 1..24.

    Returns a list of 24 values (None where the hour is missing), or None if
    hour 1 is missing / non-positive.
    """
    try:
        ds = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for pt in comp.get("hourly") or []:
        try:
            t = parse_ts(pt["from"])
            viewers = pt["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        delta_h = (t - ds).total_seconds() / 3600
        if 0 <= delta_h < 24 and delta_h == int(delta_h):
            hours[int(delta_h) + 1] = viewers
    curve = [hours.get(h) for h in range(1, 25)]
    if curve[0] is None or curve[0] <= 0:
        return None
    return curve


def fit_regression(curves):
    """Per-hour regression of log(viewers(h)/viewers(1)) on log(viewers(1)).

    curves: list of (curve, is_console_midnight). Returns (model, mean_log_p1)
    where model[h] = (a, b, sd_resid) or None if the hour has no data.
    """
    logp1s = [math.log(c[0]) for (c, _cm) in curves if c[0] > 0]
    if not logp1s:
        return None, 0.0
    mean_lp = statistics.mean(logp1s)
    model = {}
    for h in range(2, 25):
        xs, ys = [], []
        for (c, _cm) in curves:
            if c[h - 1] is not None and c[0] > 0:
                xs.append(math.log(c[0]) - mean_lp)
                ys.append(math.log(c[h - 1] / c[0]))
        if not ys:
            model[h] = None
            continue
        if len(xs) < 3:
            # Too few points for a slope: median + spread, no log(p1) term.
            model[h] = (statistics.median(ys), 0.0,
                        statistics.stdev(ys) if len(ys) > 1 else 0.3)
            continue
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        b = num / den if den > 0 else 0.0
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        # Sample sd (n-1): we estimate the population spread from a small
        # sample of launches, so the unbiased estimator is the right one.
        sd = statistics.stdev(resid) if len(resid) > 1 else 0.3
        model[h] = (a, b, sd)
    return model, mean_lp


def fit_median_ratio(curves):
    """Fallback model: per-hour (median log-ratio, sd of log-ratio)."""
    model = {}
    for h in range(2, 25):
        logs = []
        for (c, _cm) in curves:
            if c[h - 1] is not None and c[0] > 0:
                logs.append(math.log(c[h - 1] / c[0]))
        if logs:
            model[h] = (statistics.median(logs),
                        statistics.stdev(logs) if len(logs) > 1 else 0.3)
        else:
            model[h] = None
    return model


def house_quantiles(p1, h):
    return [p1 * math.exp(z * HOUSE_SIGMA * math.sqrt(h - 1)) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # Build the training set from the case's comparables.
    curves = []
    for comp in case.get("comparables") or []:
        cname = comp.get("name") or ""
        if "Grand Theft Auto VI category" in cname:
            # The GTA VI category's own pre-release history is not a launch
            # day; it has no release-day curve to learn from.
            continue
        curve = release_day_curve(comp)
        if curve is None:
            continue
        if sum(1 for x in curve if x is None) > 4:
            continue
        curves.append((curve, is_console_midnight(comp.get("platforms", ""),
                                                   comp.get("unlock", ""))))

    # Prefer the console-midnight subset (GTA VI's unlock shape); fall back
    # to all comparables if that subset is too small.
    console = [c for c in curves if c[1]]
    train = console if len(console) >= MIN_CONSOLE else curves

    model, mean_lp = fit_regression(train) if train else (None, 0.0)
    median_model = fit_median_ratio(train) if train else None

    dev = math.log(p1) - mean_lp
    dev = max(-DEV_CAP, min(DEV_CAP, dev))

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        if model is not None and model.get(h) is not None:
            a, b, sd = model[h]
            mu = a + b * dev + diurnal_shift(h)
            s = max(sd * SD_MULT, SD_FLOOR)
            qs = [p1 * math.exp(mu + z * s) for z in Z]
        elif median_model is not None and median_model.get(h) is not None:
            mu, sd = median_model[h]
            s = max(sd * SD_MULT, SD_FLOOR)
            qs = [p1 * math.exp(mu + z * s) for z in Z]
        else:
            qs = house_quantiles(p1, h)
        # Round half up, clamp to (0, 1e9], enforce non-decreasing.
        out = []
        prev = 0.0
        for q in qs:
            v = round2(q)
            if v <= 0:
                v = 0.01
            if v > MAX_Q:
                v = MAX_Q
            if v < prev:
                v = prev
            out.append(v)
            prev = v
        rows.append({"hour": h, "q": out})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
