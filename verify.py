"""Recompute every number reported in the paper from data/ and outputs/; exit 1 on any failure.
Tolerances: 0.0051 for 2-decimal percentages; 0.5% relative for death counts (posterior medians)."""
import json, pathlib, sys
import numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parent
E = json.loads((R / "results/expected_key_results.json").read_text())
fails, n = [], 0
def check(name, got, want, tol):
    global n; n += 1; ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:70s} got {got:+.4f}  expected {want:+.4f}")
    if not ok: fails.append(name)
fe = pd.read_csv(R / "outputs/fe_results.csv"); fe.index = fe.analysis + "|" + fe.term
for k, (m, lo, hi) in E["fe"].items():
    for lab, col, w in (("est", "pct", m), ("lo", "lo", lo), ("hi", "hi", hi)): check(f"FE {k} {lab}", fe.loc[k, col], w, 0.0051)
check("FE sample state-months", fe.loc["A joint heat+rain (deaths)|T", "n"], E["sample"]["state_months"], 0)
pp = pd.read_csv(R / "outputs/ppml_results.csv"); pp.index = pp.model + "|" + pp.term
for k, (m, lo, hi) in E["ppml"].items():
    for lab, col, w in (("est", "pct", m), ("lo", "lo", lo), ("hi", "hi", hi)): check(f"PPML {k} {lab}", pp.loc[k, col], w, 0.0051)
z = np.load(R / "outputs/bayes_hier_draws.npz", allow_pickle=True)   # our own output (string array of state names)
MB = z["mu_beta"]; pc = lambda v: np.percentile(100 * np.expm1(v), [50, 2.5, 97.5])
for k, v in (("heat same", MB[:, 0]), ("heat next", MB[:, 1]), ("heat net", MB[:, 0] + MB[:, 1]), ("rain net", MB[:, 2] + MB[:, 3])):
    for lab, g, w in zip(("median", "lo", "hi"), pc(v), E["bayes_national"][k]): check(f"Bayes {k} {lab} (from draws)", g, w, 0.0051)
for k, v in (("naive", MB[:, 0] * 2), ("aware", (MB[:, 0] + MB[:, 1]) * 2)):
    for lab, g, w in zip(("median", "lo", "hi"), pc(v), E["bayes_projection_2C"][k]): check(f"Bayes +2C {k} {lab}", g, w, 0.0051)
check("Bayes states in model", len(z["states"]), E["sample"]["states"], 0)
sm = pd.read_csv(R / "outputs/bayes_hier_summary.csv").set_index("quantity")
for k, q in (("same-month excess", "deaths 2014-2024, heat: same-month excess from above-normal months"), ("net", "deaths 2014-2024, heat: net (both lags), all months")):
    w = E["bayes_accounting"][k]; check(f"Bayes accounting {k} deaths", sm.loc[q, "median"], w, 0.005 * abs(w))
wx = pd.read_csv(R / "data/state_month_weather_1991_2024.csv")
check("weather units", wx.state.nunique(), E["sample"]["weather_units"], 0)
deaths = pd.read_csv(R / "data/rai_state_month_2014_2024.csv")
check("weather covers every death-panel unit", len(set(deaths.state) - set(wx.state)), 0, 0)
print(f"\n{n - len(fails)}/{n} checks passed"); sys.exit(1 if fails else 0)
