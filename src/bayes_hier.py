"""Bayesian hierarchical negative-binomial model of monthly road deaths with state-varying lagged weather effects.

  deaths_st ~ NegBin(mu_st, phi)
  log mu_st = a[state x calendar month] + d[national month]
              + b0_s T_st + b1_s T_s,t-1 + g0_s R_st + g1_s R_s,t-1
  (b0_s, b1_s, g0_s, g1_s) ~ N((B0, B1, G0, G1), diag(tau^2))   partial pooling across 22 states
T = tmax anomaly (C), R = rainfall anomaly (per 100 mm), vs the 1991-2020 normal. COVID months dropped.

Derived quantities (per posterior draw):
  * national same-month, next-month and net (b0 + b1) effects for heat and rain
  * accounting over 2014-2024: deaths 'brought forward' by hot months (same-month excess from positive T), deaths
    moved out of following months, net heat deaths; deaths averted by above-normal rain (net, both lags)
  * projection contrast for a sustained uniform warming of +1 and +2 C applied to every month: naive same-month model
    (b0 only) vs displacement-aware model (b0 + b1), expressed as % change in annual deaths
Outputs: outputs/bayes_hier_summary.csv, outputs/bayes_hier_states.csv, outputs/bayes_hier_draws.npz, log
"""
import os, pathlib, time
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)

P = pathlib.Path(__file__).resolve().parents[1]
deaths = pd.read_csv(P / "data/rai_state_month_2014_2024.csv", parse_dates=["date"])
wx = pd.read_csv(P / "data/state_month_weather_1991_2024.csv", parse_dates=["date"]).sort_values(["state", "date"])
wx["T"] = wx.tmax_mean_anom; wx["R"] = wx.rain_total_anom / 100
wx["T1"] = wx.groupby("state")["T"].shift(1); wx["R1"] = wx.groupby("state")["R"].shift(1)
d = deaths.merge(wx[["state", "date", "T", "R", "T1", "R1"]], on=["state", "date"], how="left")
COVID = {f"{y}-{m:02d}" for y, ms in ((2020, range(3, 7)), (2021, range(4, 7))) for m in ms}
d = d[~d.date.dt.strftime("%Y-%m").isin(COVID)]
size = d.groupby("state").killed.mean()
d = d[d.state.isin(size[size >= 50].index)].dropna(subset=["T", "R", "T1", "R1"]).reset_index(drop=True)
st_idx, states = pd.factorize(d.state)
sc_idx, _ = pd.factorize(d.state + "_" + d.date.dt.month.astype(str))
ym_idx, yms = pd.factorize(d.date.dt.strftime("%Y-%m"))
S, NSC, NYM = len(states), sc_idx.max() + 1, ym_idx.max() + 1
X = jnp.array(d[["T", "T1", "R", "R1"]].values)
y = jnp.array(d.killed.values.astype(int))
st = jnp.array(st_idx); sc = jnp.array(sc_idx); ym = jnp.array(ym_idx)
ybar = d.groupby(sc_idx).killed.mean().reindex(range(NSC)).values
print(f"{S} states, {len(d)} state-months, {NSC} state x calendar-month cells, {NYM} months")


def model(X, y=None):
    a = numpyro.sample("a", dist.Normal(jnp.log(jnp.array(ybar)), 0.5))
    dr = numpyro.sample("d_raw", dist.Normal(0, 0.3).expand([NYM])); dm = dr - dr.mean()
    mu_beta = numpyro.sample("mu_beta", dist.Normal(0, 0.05).expand([4]))          # national effects (log scale)
    tau = numpyro.sample("tau", dist.HalfNormal(0.02).expand([4]))
    z = numpyro.sample("z", dist.Normal(0, 1).expand([S, 4]))
    beta = numpyro.deterministic("beta", mu_beta + tau * z)                          # S x 4 state effects
    phi = numpyro.sample("phi", dist.Gamma(2.0, 0.01))
    eta = a[sc] + dm[ym] + jnp.sum(beta[st] * X, axis=1)
    numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(eta), phi), obs=y)


t0 = time.time()
NW, ND = int(os.environ.get("NWARM", 1000)), int(os.environ.get("NDRAW", 1000))
mcmc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=9), num_warmup=NW, num_samples=ND, num_chains=4, progress_bar=False)
mcmc.run(jax.random.PRNGKey(20261109), X, y, extra_fields=("diverging",))
sm = jax.block_until_ready(mcmc.get_samples())
print(f"sampled in {time.time() - t0:.0f}s; divergences {int(mcmc.get_extra_fields()['diverging'].sum())}")
from numpyro.diagnostics import summary
g = mcmc.get_samples(group_by_chain=True)
sumry = summary({"mu_beta": g["mu_beta"], "tau": g["tau"], "phi": g["phi"]}, prob=0.95)
for k, v in sumry.items():
    print(k, "mean", np.round(v["mean"], 4), "| r_hat", np.round(v["r_hat"], 3), "| n_eff", np.round(v["n_eff"]).astype(int))

beta = np.asarray(sm["beta"]); MB = np.asarray(sm["mu_beta"])                    # draws x S x 4, draws x 4
np.savez_compressed(P / "outputs/bayes_hier_draws.npz", beta=beta.astype(np.float32), mu_beta=MB, states=np.array(states))
pc = lambda v: np.percentile(100 * np.expm1(v), [50, 2.5, 97.5])
rows = []
for lab, v in (("heat same month (+1 C)", MB[:, 0]), ("heat next month (+1 C)", MB[:, 1]), ("heat net 0-1 month", MB[:, 0] + MB[:, 1]),
               ("rain same month (+100 mm)", MB[:, 2]), ("rain next month (+100 mm)", MB[:, 3]), ("rain net 0-1 month", MB[:, 2] + MB[:, 3])):
    m_, l_, h_ = pc(v); rows.append(dict(quantity=lab, median=m_, lo=l_, hi=h_, p_positive=float((v > 0).mean())))

# ---- accounting over the observed period (expected deaths with vs without the anomaly terms) ----
a = np.asarray(sm["a"]); dr = np.asarray(sm["d_raw"]); dm = dr - dr.mean(1, keepdims=True)
base = a[:, sc_idx] + dm[:, ym_idx]                                                  # draws x N
Xn = np.asarray(X); B = beta[:, st_idx, :]                                          # draws x N x 4
lin = lambda cols: np.einsum("dnk,nk->dn", B[:, :, cols], Xn[:, cols])
mu_obs = np.exp(base + lin([0, 1, 2, 3]))
acc = {"heat: same-month excess from above-normal months": np.exp(base + lin([1, 2, 3])) * 0 + (np.exp(base + lin([0, 1, 2, 3])) - np.exp(base + lin([1, 2, 3]))) * (Xn[:, 0] > 0),
       "heat: net (both lags), all months": mu_obs - np.exp(base + lin([2, 3])),
       "rain: net (both lags), all months": mu_obs - np.exp(base + lin([0, 1]))}
yrs = (len(d) / S) / 12
for lab, arr in acc.items():
    tot = arr.sum(1)
    m_, l_, h_ = np.percentile(tot, [50, 2.5, 97.5])
    rows.append(dict(quantity=f"deaths 2014-2024, {lab}", median=m_, lo=l_, hi=h_, p_positive=float((tot > 0).mean())))

# ---- projection contrast: sustained uniform warming applied to T and T1 (anomalies vs the 1991-2020 normal) ----
for dT in (1.0, 2.0):
    naive = 100 * np.expm1(MB[:, 0] * dT); aware = 100 * np.expm1((MB[:, 0] + MB[:, 1]) * dT)
    for lab, v in (("naive same-month model", naive), ("displacement-aware model", aware)):
        m_, l_, h_ = np.percentile(v, [50, 2.5, 97.5])
        rows.append(dict(quantity=f"projection +{dT:.0f} C sustained: {lab} (% annual deaths)", median=m_, lo=l_, hi=h_, p_positive=float((v > 0).mean())))
res = pd.DataFrame(rows)
res.to_csv(P / "outputs/bayes_hier_summary.csv", index=False)
pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))
stt = pd.DataFrame({"state": states,
                    **{f"heat_net_{q}": np.percentile(100 * np.expm1(beta[:, :, 0] + beta[:, :, 1]), p, 0) for q, p in (("med", 50), ("lo", 2.5), ("hi", 97.5))},
                    **{f"rain_net_{q}": np.percentile(100 * np.expm1(beta[:, :, 2] + beta[:, :, 3]), p, 0) for q, p in (("med", 50), ("lo", 2.5), ("hi", 97.5))}})
stt.to_csv(P / "outputs/bayes_hier_states.csv", index=False)
print(stt.round(2).to_string(index=False))
