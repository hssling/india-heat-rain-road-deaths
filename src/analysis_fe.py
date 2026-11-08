"""Heat, rain and monthly road deaths in Indian states, 2014-2024: fixed-effects analyses.

Identification: log outcome_st = state x calendar-month FE + national month FE + weather anomalies + e.
Only a state's deviation from its own seasonal normal, relative to other states in the same month, identifies the
effect. COVID months dropped. Units: states averaging >= 50 deaths/month (22). SEs clustered by state.
Exposures (population-weighted IMD, anomalies vs the 1991-2020 state x calendar-month normal):
  tmax_anom (C), rain_anom (per 100 mm), hot35/hot40 anomalies (days), heavy-rain-day anomaly.

Analyses (outputs/fe_results.csv):
  A  joint heat + rain, deaths
  B  falsification: next month's anomalies (leads) added; lag (previous month) added
  C  relative vs absolute heat: tmax anomaly vs days >= 35 C / >= 40 C (anomalies), and anomaly x hot-climate state
  D  outcomes: deaths, accidents, injuries, case fatality (deaths per accident)
  E  season: pre-monsoon (Mar-Jun), monsoon (Jul-Sep), post-monsoon/winter (Oct-Feb)
  F  binned dose-response for tmax anomaly holding rain anomaly fixed
"""
import pathlib
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

P = pathlib.Path(__file__).resolve().parents[1]
deaths = pd.read_csv(P / "data/rai_state_month_2014_2024.csv", parse_dates=["date"])
wx = pd.read_csv(P / "data/state_month_weather_1991_2024.csv", parse_dates=["date"]).sort_values(["state", "date"])
for v in ("tmax_mean_anom", "rain_total_anom"):
    wx[f"{v}_lead"] = wx.groupby("state")[v].shift(-1)
    wx[f"{v}_lag"] = wx.groupby("state")[v].shift(1)
d = deaths.merge(wx, on=["state", "date"], how="left")
COVID = {f"{y}-{m:02d}" for y, ms in ((2020, range(3, 7)), (2021, range(4, 7))) for m in ms}
d = d[~d.date.dt.strftime("%Y-%m").isin(COVID)]
size = d.groupby("state").killed.mean()
d = d[d.state.isin(size[size >= 50].index)].copy()
d = d[(d.killed > 0) & (d.acc > 0) & (d.inj > 0)]
d["sc"] = d.state + "_" + d.date.dt.month.astype(str); d["ym"] = d.date.dt.strftime("%Y-%m")
d["T"] = d.tmax_mean_anom; d["R"] = d.rain_total_anom / 100
d["T_lead"] = d.tmax_mean_anom_lead; d["R_lead"] = d.rain_total_anom_lead / 100
d["T_lag"] = d.tmax_mean_anom_lag; d["R_lag"] = d.rain_total_anom_lag / 100
d["H35"] = d.hot35_anom; d["H40"] = d.hot40_anom; d["HV"] = d.heavy_days_anom
for v, src in (("ly", "killed"), ("la", "acc"), ("li", "inj")): d[v] = np.log(d[src])
d["lcf"] = np.log(d.killed / d.acc)
clim = wx[(wx.date.dt.year >= 1991) & (wx.date.dt.year <= 2020)].groupby("state").tmax_mean.mean()
d["hotclim"] = d.state.map(clim >= clim.reindex(d.state.unique()).median()).astype(int)
d["season"] = np.select([d.date.dt.month.isin([3, 4, 5, 6]), d.date.dt.month.isin([7, 8, 9])], ["premonsoon", "monsoon"], "winter")
print(f"{d.state.nunique()} states, {len(d)} state-months; hot-climate states: {sorted(d[d.hotclim == 1].state.unique())}")

FE = "C(sc) + C(ym)"
rows = []


def fit(label, formula, data=d, keep=None):
    m = smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data.state)[0]})
    for term in [t for t in m.params.index if (not t.startswith("C(") or t.startswith("C(Tbin)")) and t != "Intercept"]:
        if keep and term not in keep: continue
        lo, hi = m.conf_int().loc[term]
        rows.append(dict(analysis=label, term=term, pct=100 * np.expm1(m.params[term]), lo=100 * np.expm1(lo),
                         hi=100 * np.expm1(hi), p=m.pvalues[term], n=int(m.nobs)))
    return m


fit("A joint heat+rain (deaths)", f"ly ~ {FE} + T + R")
fit("B falsification: + leads", f"ly ~ {FE} + T + R + T_lead + R_lead", d.dropna(subset=["T_lead", "R_lead"]))
fit("B + lags", f"ly ~ {FE} + T + R + T_lag + R_lag", d.dropna(subset=["T_lag", "R_lag"]))
mlag = fit("B2 lags 0-2 (displacement)", f"ly ~ {FE} + T + T_lag + T_lag2 + R + R_lag", d.assign(T_lag2=d.groupby("state")["T_lag"].shift(1)).dropna(subset=["T_lag", "R_lag"]).assign(T_lag2=lambda x: x.T_lag2.fillna(0)))
# cumulative (net) effect of a +1 C anomaly over the month and the next month, with its CI
mc = smf.ols(f"ly ~ {FE} + T + T_lag + R + R_lag", d.dropna(subset=["T_lag", "R_lag"])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.dropna(subset=["T_lag", "R_lag"]).state)[0]})
for lab, expr in (("heat, cumulative 0-1 month", "T + T_lag = 0"), ("rain, cumulative 0-1 month", "R + R_lag = 0")):
    tt = mc.t_test(expr); est = float(np.atleast_1d(tt.effect)[0]); lo, hi = np.atleast_2d(tt.conf_int())[0]
    rows.append(dict(analysis="B3 cumulative", term=lab, pct=100 * np.expm1(est), lo=100 * np.expm1(lo), hi=100 * np.expm1(hi), p=float(tt.pvalue), n=int(mc.nobs)))
fit("C absolute heat: days >=35 C", f"ly ~ {FE} + H35 + R")
fit("C absolute heat: days >=40 C", f"ly ~ {FE} + H40 + R")
fit("C relative heat x hot-climate state", f"ly ~ {FE} + T + T:hotclim + R")
for v, lab in (("la", "accidents"), ("li", "injuries"), ("lcf", "case fatality (deaths/accident)")):
    fit(f"D outcome: {lab}", f"{v} ~ {FE} + T + R")
for s in ("premonsoon", "monsoon", "winter"):
    fit(f"E season: {s}", f"ly ~ {FE} + T + R", d[d.season == s])
d["Tbin"] = pd.cut(d["T"], [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf], labels=["<=-1.5", "-1.5..-0.5", "ref", "0.5..1.5", ">1.5"])
d["Tbin"] = d.Tbin.cat.reorder_categories(["ref", "<=-1.5", "-1.5..-0.5", "0.5..1.5", ">1.5"])
fit("F tmax-anomaly bins (rain held fixed)", f"ly ~ {FE} + C(Tbin) + R")
for lab in ("<=-1.5", "-1.5..-0.5", "0.5..1.5", ">1.5"):
    pass
res = pd.DataFrame(rows)
res["term"] = res.term.str.replace(r"C\(Tbin\)\[T\.(.*)\]", r"T bin \1", regex=True)
res.to_csv(P / "outputs/fe_results.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\ncorrelation of T and R anomalies:", round(d[["T", "R"]].corr().iloc[0, 1], 3))
