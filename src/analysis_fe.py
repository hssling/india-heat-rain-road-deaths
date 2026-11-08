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

rows = []
Z = 1.959963984540054                                              # normal quantile, as in statsmodels' cluster CIs


def fwl(ycol, xcols, data):
    """OLS of ycol on xcols with state x calendar-month and national month fixed effects, by Frisch-Waugh-Lovell:
    outcome and regressors are residualised on an orthonormal (SVD) basis of the fixed-effect space, then the
    cluster-robust (state) covariance is computed for the regressors only. Point estimates equal the full dummy
    regression; the small-sample factor G/(G-1)*(N-1)/(N-K) uses K = all dummy columns (1 + (S-1) + (M-1) + p),
    matching statsmodels' formula fit. Avoids the rank-deficient full covariance (~390 dummies, 22 clusters),
    whose clustered CIs became NaN under statsmodels 0.15 / pandas 3."""
    data = data.dropna(subset=[ycol] + xcols)
    dummies = pd.get_dummies(data[["sc", "ym"]].astype(str)).astype(float).values
    Dc = np.column_stack([np.ones(len(data)), dummies])
    U, s, _ = np.linalg.svd(Dc, full_matrices=False)
    Q = U[:, s > s.max() * max(Dc.shape) * np.finfo(float).eps]
    res = lambda v: v - Q @ (Q.T @ v)
    y = res(data[ycol].to_numpy(float)); X = np.column_stack([res(data[c].to_numpy(float)) for c in xcols])
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ (X.T @ y); e = y - X @ b
    g = pd.factorize(data.state)[0]; G = int(g.max()) + 1
    meat = sum(np.outer(X[g == k].T @ e[g == k], X[g == k].T @ e[g == k]) for k in range(G))
    N, K = len(y), 1 + (data.sc.nunique() - 1) + (data.ym.nunique() - 1) + len(xcols)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((N - 1) / (N - K))
    return b, V, N


def add(label, term, est, var, n):
    se = np.sqrt(var); from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(est / se) / sqrt(2))))
    rows.append(dict(analysis=label, term=term, pct=100 * np.expm1(est), lo=100 * np.expm1(est - Z * se),
                     hi=100 * np.expm1(est + Z * se), p=p, n=n))


def fit(label, ycol, xcols, data=d, names=None):
    b, V, n = fwl(ycol, xcols, data)
    for i, c in enumerate(xcols):
        add(label, (names or xcols)[i], b[i], V[i, i], n)
    return b, V, n


d["T:hotclim"] = d["T"] * d.hotclim
d["Tbin"] = pd.cut(d["T"], [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf], labels=["<=-1.5", "-1.5..-0.5", "ref", "0.5..1.5", ">1.5"])
BINS = ["<=-1.5", "-1.5..-0.5", "0.5..1.5", ">1.5"]
for lab in BINS: d[f"T bin {lab}"] = (d.Tbin == lab).astype(float)
lagged = d.dropna(subset=["T_lag", "R_lag"]).copy()
lagged["T_lag2"] = lagged.groupby("state")["T_lag"].shift(1).fillna(0)

fit("A joint heat+rain (deaths)", "ly", ["T", "R"])
fit("B falsification: + leads", "ly", ["T", "R", "T_lead", "R_lead"], d.dropna(subset=["T_lead", "R_lead"]))
fit("B + lags", "ly", ["T", "R", "T_lag", "R_lag"], lagged)
fit("B2 lags 0-2 (displacement)", "ly", ["T", "T_lag", "T_lag2", "R", "R_lag"], lagged)
b, V, n = fwl("ly", ["T", "T_lag", "R", "R_lag"], lagged)          # cumulative (net) effects over the month and the next
for lab, w in (("heat, cumulative 0-1 month", np.array([1, 1, 0, 0])), ("rain, cumulative 0-1 month", np.array([0, 0, 1, 1]))):
    add("B3 cumulative", lab, float(w @ b), float(w @ V @ w), n)
fit("C absolute heat: days >=35 C", "ly", ["H35", "R"])
fit("C absolute heat: days >=40 C", "ly", ["H40", "R"])
fit("C relative heat x hot-climate state", "ly", ["T", "T:hotclim", "R"])
for v, lab in (("la", "accidents"), ("li", "injuries"), ("lcf", "case fatality (deaths/accident)")):
    fit(f"D outcome: {lab}", v, ["T", "R"])
for s in ("premonsoon", "monsoon", "winter"):
    fit(f"E season: {s}", "ly", ["T", "R"], d[d.season == s])
fit("F tmax-anomaly bins (rain held fixed)", "ly", [f"T bin {lab}" for lab in BINS] + ["R"])
res = pd.DataFrame(rows)
res.to_csv(P / "outputs/fe_results.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\ncorrelation of T and R anomalies:", round(d[["T", "R"]].corr().iloc[0, 1], 3))
