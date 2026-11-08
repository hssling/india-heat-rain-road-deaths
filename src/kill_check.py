"""Kill check: do weather anomalies move monthly road deaths once seasonality and common shocks are removed?

log(deaths_st) = state x calendar-month FE + national month FE + b'anomalies_st + e   (2014-2024, COVID months
dropped, units averaging >= 50 deaths/month). Identification comes only from a state's deviation from its own
seasonal norm relative to other states in the same month. SEs clustered by state.
Exposures: population-weighted IMD anomalies vs the 1991-2020 normal (src/build_exposures.py).
Kill rule: no anomaly coefficient distinguishable from zero and no dose-response across hot-day bins -> stop.
"""
import pathlib
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

P = pathlib.Path(__file__).resolve().parents[1]
deaths = pd.read_csv(P / "data/rai_state_month_2014_2024.csv", parse_dates=["date"])
wx = pd.read_csv(P / "data/state_month_weather_1991_2024.csv", parse_dates=["date"])
d = deaths.merge(wx, on=["state", "date"], how="left")
COVID = {f"{y}-{m:02d}" for y, ms in ((2020, range(3, 7)), (2021, range(4, 7))) for m in ms}
d = d[~d.date.dt.strftime("%Y-%m").isin(COVID)]
size = d.groupby("state").killed.mean()
d = d[d.state.isin(size[size >= 50].index) & (d.killed > 0)].copy()
d["ly"] = np.log(d.killed)
d["sc"] = d.state + "_" + d.date.dt.month.astype(str)
d["ym"] = d.date.dt.strftime("%Y-%m")
print(f"{d.state.nunique()} states, {len(d)} state-months")
d["tmax_anom_c"] = d.tmax_mean_anom
d["rain100_anom"] = d.rain_total_anom / 100
specs = {
    "temperature (per +1 C tmax anomaly)": "ly ~ C(sc) + C(ym) + tmax_anom_c",
    "rain (per +100 mm anomaly)": "ly ~ C(sc) + C(ym) + rain100_anom",
    "heavy-rain days (per +1 day)": "ly ~ C(sc) + C(ym) + heavy_days_anom",
    "hot days >=35 C (per +1 day)": "ly ~ C(sc) + C(ym) + hot35_anom",
    "joint": "ly ~ C(sc) + C(ym) + tmax_anom_c + rain100_anom + heavy_days_anom",
}
rows = []
for lab, f in specs.items():
    m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.state)[0]})
    for term in [t for t in m.params.index if not t.startswith("C(") and t != "Intercept"]:
        lo, hi = m.conf_int().loc[term]
        rows.append(dict(model=lab, term=term, pct=100 * np.expm1(m.params[term]), lo=100 * np.expm1(lo),
                         hi=100 * np.expm1(hi), p=m.pvalues[term]))
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
# dose-response: bins of the tmax anomaly
d["tbin"] = pd.cut(d.tmax_anom_c, [-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf],
                   labels=["<-2", "-2..-1", "-1..-0.5", "ref", "0.5..1", "1..2", ">2"])
d["tbin"] = d.tbin.cat.reorder_categories(["ref", "<-2", "-2..-1", "-1..-0.5", "0.5..1", "1..2", ">2"])
m = smf.ols("ly ~ C(sc) + C(ym) + C(tbin)", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.state)[0]})
print("\ntmax-anomaly bins (vs -0.5..+0.5 C):")
for t in [t for t in m.params.index if t.startswith("C(tbin)")]:
    lab = t.split("[T.")[1].rstrip("]")
    print(f"  {lab:10s} {100*np.expm1(m.params[t]):+6.2f}%  p={m.pvalues[t]:.3f}  n={int((d.tbin == lab).sum())}")
res.to_csv(P / "outputs_kill_check.csv", index=False)
