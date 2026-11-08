"""Robustness: Poisson pseudo-maximum-likelihood (count model) with the same fixed effects as analysis_fe.py.

deaths_st ~ exp(state x calendar-month FE + national month FE + b'X), cluster-robust SEs by state.
Checks that the log-OLS findings (rain lowers deaths; heat effect displaced across months) hold for counts.
Output: outputs/ppml_results.csv
"""
import pathlib
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

P = pathlib.Path(__file__).resolve().parents[1]
deaths = pd.read_csv(P / "data/rai_state_month_2014_2024.csv", parse_dates=["date"])
wx = pd.read_csv(P / "data/state_month_weather_1991_2024.csv", parse_dates=["date"]).sort_values(["state", "date"])
wx["T_lag"] = wx.groupby("state").tmax_mean_anom.shift(1); wx["R_lag"] = wx.groupby("state").rain_total_anom.shift(1) / 100
wx["T_lead"] = wx.groupby("state").tmax_mean_anom.shift(-1); wx["R_lead"] = wx.groupby("state").rain_total_anom.shift(-1) / 100
d = deaths.merge(wx, on=["state", "date"], how="left")
COVID = {f"{y}-{m:02d}" for y, ms in ((2020, range(3, 7)), (2021, range(4, 7))) for m in ms}
d = d[~d.date.dt.strftime("%Y-%m").isin(COVID)]
size = d.groupby("state").killed.mean()
d = d[d.state.isin(size[size >= 50].index)].dropna(subset=["T_lag", "R_lag", "T_lead", "R_lead"]).copy()
d["T"] = d.tmax_mean_anom; d["R"] = d.rain_total_anom / 100
d["sc"] = d.state + "_" + d.date.dt.month.astype(str); d["ym"] = d.date.dt.strftime("%Y-%m")
rows = []
for lab, rhs in (("joint", "T + R"), ("lags", "T + T_lag + R + R_lag"), ("leads (falsification)", "T + R + T_lead + R_lead")):
    m = smf.glm(f"killed ~ C(sc) + C(ym) + {rhs}", d, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.state)[0]})
    for term in rhs.replace(" ", "").split("+"):
        lo, hi = m.conf_int().loc[term]
        rows.append(dict(model=lab, term=term, pct=100 * np.expm1(m.params[term]), lo=100 * np.expm1(lo), hi=100 * np.expm1(hi), p=m.pvalues[term]))
    if lab == "lags":
        for nm, expr in (("heat cumulative 0-1", "T + T_lag = 0"), ("rain cumulative 0-1", "R + R_lag = 0")):
            tt = m.t_test(expr); est = float(np.atleast_1d(tt.effect)[0]); lo, hi = np.atleast_2d(tt.conf_int())[0]
            rows.append(dict(model=lab, term=nm, pct=100 * np.expm1(est), lo=100 * np.expm1(lo), hi=100 * np.expm1(hi), p=float(tt.pvalue)))
res = pd.DataFrame(rows)
res.to_csv(P / "outputs/ppml_results.csv", index=False)
print(f"{d.state.nunique()} states, {len(d)} state-months")
print(res.round(3).to_string(index=False))
