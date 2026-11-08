"""Prior sensitivity for the Bayesian hierarchical model (bayes_hier.py, Eq. 1).

Refits Eq. (1) under alternative priors for the national effects (mu_beta) and the between-state SDs (tau),
keeping everything else identical (data, fixed effects, sampler settings, seed):
  base     mu_beta ~ N(0, 0.05),  tau ~ HalfNormal(0.02)     (main analysis)
  wide     mu_beta ~ N(0, 0.20),  tau ~ HalfNormal(0.10)     (4-5x wider on both)
  vague    mu_beta ~ N(0, 1.00),  tau ~ HalfNormal(0.25)
Reports national same-month / next-month / net effects (% per +1 C, per +100 mm), posterior tau medians, max r_hat,
min n_eff and divergences. Output: outputs/bayes_prior_sensitivity.csv
"""
import os, pathlib
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary
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
ym_idx, _ = pd.factorize(d.date.dt.strftime("%Y-%m"))
S, NSC, NYM = len(states), sc_idx.max() + 1, ym_idx.max() + 1
X = jnp.array(d[["T", "T1", "R", "R1"]].values); y = jnp.array(d.killed.values.astype(int))
st = jnp.array(st_idx); sc = jnp.array(sc_idx); ym = jnp.array(ym_idx)
ybar = d.groupby(sc_idx).killed.mean().reindex(range(NSC)).values
assert len(d) == 2750 and S == 22, (len(d), S)


def make_model(s_mu, s_tau):
    def model(X, y=None):
        a = numpyro.sample("a", dist.Normal(jnp.log(jnp.array(ybar)), 0.5))
        dr = numpyro.sample("d_raw", dist.Normal(0, 0.3).expand([NYM])); dm = dr - dr.mean()
        mu_beta = numpyro.sample("mu_beta", dist.Normal(0, s_mu).expand([4]))
        tau = numpyro.sample("tau", dist.HalfNormal(s_tau).expand([4]))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([S, 4]))
        beta = numpyro.deterministic("beta", mu_beta + tau * z)
        phi = numpyro.sample("phi", dist.Gamma(2.0, 0.01))
        eta = a[sc] + dm[ym] + jnp.sum(beta[st] * X, axis=1)
        numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(eta), phi), obs=y)
    return model


pc = lambda v: np.percentile(100 * np.expm1(v), [50, 2.5, 97.5])
rows = []
for name, s_mu, s_tau in (("base", 0.05, 0.02), ("wide", 0.20, 0.10), ("vague", 1.00, 0.25)):
    mcmc = MCMC(NUTS(make_model(s_mu, s_tau), target_accept_prob=0.9, max_tree_depth=9),
                num_warmup=1000, num_samples=1000, num_chains=4, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(20261109), X, y, extra_fields=("diverging",))
    g = mcmc.get_samples(group_by_chain=True)
    sm = summary({"mu_beta": g["mu_beta"], "tau": g["tau"], "phi": g["phi"]}, prob=0.95)
    rhat = max(float(np.max(v["r_hat"])) for v in sm.values()); neff = min(float(np.min(v["n_eff"])) for v in sm.values())
    ndiv = int(mcmc.get_extra_fields()["diverging"].sum())
    MB = np.asarray(mcmc.get_samples()["mu_beta"]); TAU = np.asarray(mcmc.get_samples()["tau"])
    beta = np.asarray(mcmc.get_samples()["beta"])
    rn = np.median(100 * np.expm1(beta[:, :, 2] + beta[:, :, 3]), 0)
    for lab, v in (("heat same month", MB[:, 0]), ("heat next month", MB[:, 1]), ("heat net 0-1", MB[:, 0] + MB[:, 1]),
                   ("rain same month", MB[:, 2]), ("rain next month", MB[:, 3]), ("rain net 0-1", MB[:, 2] + MB[:, 3])):
        m_, l_, h_ = pc(v)
        rows.append(dict(prior=name, quantity=lab, median=m_, lo=l_, hi=h_))
    for k, lab in enumerate(("tau heat same", "tau heat next", "tau rain same", "tau rain next")):
        rows.append(dict(prior=name, quantity=lab, median=float(np.median(TAU[:, k])),
                         lo=float(np.percentile(TAU[:, k], 2.5)), hi=float(np.percentile(TAU[:, k], 97.5))))
    rows.append(dict(prior=name, quantity="state rain-net median range", median=np.nan, lo=float(rn.min()), hi=float(rn.max())))
    rows.append(dict(prior=name, quantity="diagnostics: max r_hat / min n_eff / divergences", median=rhat, lo=neff, hi=ndiv))
    print(name, f"r_hat {rhat:.3f} n_eff {neff:.0f} div {ndiv}", flush=True)
res = pd.DataFrame(rows)
res.to_csv(P / "outputs/bayes_prior_sensitivity.csv", index=False)
pd.set_option("display.width", 200); print(res.round(4).to_string(index=False))
