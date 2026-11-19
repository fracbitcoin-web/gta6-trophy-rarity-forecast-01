"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model (own design, standard library only, deterministic):
  * Split each trophy's observed rate into earners E and players tracked N,
    rate = 100 * E / N.  N only grows (tracked players join), E only grows
    (players earn trophies); a refresh of Exophase's page moves both at once.
  * Fit log N and, per trophy, log E against log(1 + hours since the first
    read) by least squares over the distinct reads (identical consecutive
    reads carry no information and are dropped).  Extrapolate both to the
    target and divide: the median forecast rate is 100 * E_target / N_target.
    This separates real progression (E rising) from dilution (N rising
    faster than E), which is what moves rates over a launch week.
  * With fewer than three distinct reads, or a flat / non-positive fit, the
    counts stay where the last read put them (persistence in counts).
  * Spread on the logit scale: a random-walk term growing with the horizon,
    plus the residuals of the two log fits, plus an allowance for refresh
    staleness (the answer is the page as of its last refresh, and refreshes
    land hours apart).  The lower side is widened a little more than the
    upper side: dilution and staleness both push the shown rate down.
  * A trophy every tracked player holds (100%) is forecast at 100% flat:
    it survived a sixfold influx of tracked players on launch night.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

WALK_SIGMA = 0.5        # logit units per sqrt(day), random-walk spread
STALE_SIGMA = 0.30      # logit allowance for refresh staleness, no jumps seen
DOWN_WIDEN = 1.25       # extra width on the low side (dilution risk)
UP_WIDEN = 1.0
MAX_N_GROWTH = 40.0     # cap on extrapolated tracked-player growth
MAX_E_GROWTH = 30.0     # cap on extrapolated earner growth


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_of_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg > 40.0:
        return 99.99
    if lg < -40.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def least_squares(xs, ys):
    """Return (slope, intercept, residual_std) of y ~ a + b*x, or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        var = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return slope, intercept, resid


def distinct_reads(case):
    """Collapse history to reads that changed something, with hours since first."""
    history = case.get("history") or []
    if not history:
        return [], None
    t0 = parse_time(history[0]["at"])
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        hours = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
        states.append((max(hours, 0.0), read.get("players_tracked"), rates))
    return states, t0


def forecast_case(case):
    trophies = case["trophies"]
    horizon_days = max(case["horizon_minutes"], 0) / 1440.0
    target = parse_time(case["target"])
    states, t0 = distinct_reads(case)
    if not states:
        return {"trophies": [
            {"id": t["id"], "q": [1.0, 1.0, 1.0, 1.0, 1.0]} for t in trophies]}

    last_hours, last_n, last_rates = states[-1]
    target_hours = max((target - t0).total_seconds() / 3600.0, last_hours)
    x_target = math.log1p(target_hours)

    # Trend fits use only the most recent day of reads: the launch-night
    # transient (everyone earning the first trophies at once) inflates a
    # whole-history slope and would push saturating trophies past 100%.
    # With fewer than three distinct reads in the window, use all of them.
    window = [s for s in states if s[0] >= last_hours - 24.0]
    fit_states = window if len(window) >= 3 else states
    xs = [math.log1p(h) for h, _, _ in fit_states]

    # --- tracked players N at the target ---
    n_fit = None
    if len(fit_states) >= 3 and len({n for _, n, _ in fit_states}) > 1:
        n_fit = least_squares(xs, [math.log(max(n, 1)) for _, n, _ in fit_states])
    if n_fit is not None and n_fit[0] > 0:
        log_n_pred = n_fit[1] + n_fit[0] * x_target
        n_pred = math.exp(min(log_n_pred, math.log(last_n) + math.log(MAX_N_GROWTH)))
        n_pred = max(n_pred, float(last_n))
        n_resid = n_fit[2]
    else:
        n_pred = float(last_n)
        n_resid = 0.0

    # Typical refresh jump size, for the staleness allowance.
    jumps = []
    for (_, n0, _), (_, n1, _) in zip(states, states[1:]):
        if n0 and n1 and n1 > n0:
            jumps.append(math.log(n1 / n0))
    if jumps:
        jumps.sort()
        stale = 0.30 * jumps[len(jumps) // 2]
    else:
        stale = STALE_SIGMA

    walk = WALK_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is None:
            last_rate = 1.0
        if last_rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue

        # --- earners E of this trophy at the target ---
        e_series = []
        for h, n, rates in fit_states:
            r = rates[i] if i < len(rates) else None
            if r is None or n is None:
                continue
            e_series.append((math.log1p(h), max(r * n / 100.0, 0.5)))
        e_last = max(last_rate * last_n / 100.0, 0.5)
        e_fit = None
        if len(e_series) >= 3 and len({e for _, e in e_series}) > 1:
            e_fit = least_squares([x for x, _ in e_series],
                                  [math.log(e) for _, e in e_series])
        if e_fit is not None and e_fit[0] > 0:
            log_e_pred = e_fit[1] + e_fit[0] * x_target
            e_pred = math.exp(min(log_e_pred, math.log(e_last) + math.log(MAX_E_GROWTH)))
            e_pred = max(e_pred, e_last)
            e_resid = e_fit[2]
        else:
            e_pred = e_last
            e_resid = 0.0
        e_pred = min(e_pred, n_pred)

        median_rate = clamp_pct(100.0 * e_pred / n_pred)
        centre = logit_of_pct(median_rate)
        sigma = math.sqrt(walk * walk + e_resid * e_resid
                          + n_resid * n_resid + stale * stale)
        quantiles = []
        for z in Z:
            width = DOWN_WIDEN if z < 0 else UP_WIDEN
            quantiles.append(round4(inv_logit_pct(centre + z * sigma * width)))
        quantiles = [min(99.99, max(0.01, q)) for q in quantiles]
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
