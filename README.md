# Temperature, rainfall and road traffic deaths in India

This repository holds data and code for an analysis of monthly road traffic deaths in 22 Indian states, 2014–2024 (2,750 state-months; 1.61 million deaths). Deaths are linked to population-weighted temperature and rainfall anomalies from India Meteorological Department gridded data, measured relative to the 1991–2020 normal. Effects are identified within states, after removing each state's seasonal pattern and national month-to-month shocks.

[![reproduce-and-verify](https://github.com/hssling/india-heat-rain-road-deaths/actions/workflows/verify.yml/badge.svg)](https://github.com/hssling/india-heat-rain-road-deaths/actions/workflows/verify.yml)

**Status.** The accompanying manuscript has been submitted for publication. Please cite this archive (`CITATION.cff`) until the article is out.

## Main findings: hotter months displace, wetter months reduce

| Bayesian hierarchical model, national effect | Same month | Next month | Net (0–1 month) |
|---|---|---|---|
| Temperature anomaly +1 °C | +1.26% (0.44 to 2.11) | −0.89% (−1.73 to −0.08) | **+0.36% (−0.60 to 1.32)** |
| Rainfall anomaly +100 mm | −1.23% (−2.44 to 0.06) | −1.04% (−2.20 to 0.18) | **−2.27% (−3.92 to −0.54)** |

- **Most heat deaths were brought forward, not added.** Hotter-than-normal months produced about 9,850 same-month excess deaths over 2014–2024. Net of the following month's deficit, only about 1,100 remained, so roughly 89% were displaced.
- **The falsification test passes.** Next month's weather does not predict current deaths.
- **Relative heat matters; absolute heat does not.** Days at or above 35 °C or 40 °C show no effect, the anomaly effect is smaller in hot-climate states, and it is concentrated in the monsoon.
- **More crashes, not deadlier ones.** Heat raises accidents and injuries but leaves case fatality unchanged.
- **Naive projections overstate the risk.** For a sustained +2 °C warming, the same-month-only model gives +2.5% annual deaths. The displacement-aware model gives +0.7% (−1.2 to +2.7), a 3.6-fold overstatement.
- **The results hold across models.** Fixed-effects (log-OLS) and Poisson pseudo-maximum-likelihood models agree (`outputs/fe_results.csv`, `outputs/ppml_results.csv`).

## Reproduce

```bash
pip install -r requirements.txt
python verify.py          # recomputes 84 reported numbers from data/ and outputs/ (seconds)
python run_pipeline.py    # re-runs the fixed-effects and Poisson analyses, the figure, then verify
```

Optional steps:

- **Refit the Bayesian model** (NumPyro, about 2 minutes): `pip install -r requirements-bayes.txt`, then `python run_pipeline.py --bayes`.
- **Rebuild the weather exposures** (`pip install -r requirements-build.txt`): run `python run_pipeline.py --build`. This downloads about 0.9 GB of IMD gridded data with [imdlib](https://doi.org/10.1016/j.envsoft.2023.105869).
  - It also needs a Census 2011 district boundary file at `data/raw/districts_v2.geojson`, which is not redistributed here because its licence is not clearly open.
  - Any district boundary file with Census 2011 district codes (`dt_code`), district names, state names (`st_nm`) and a `year` field works. For example, see [DataMeet](https://github.com/datameet/maps).

GitHub Actions re-runs the fixed-effects and Poisson analyses on Linux and runs `verify.py` on every push.

## Contents

| Path | Contents |
|---|---|
| `data/rai_state_month_2014_2024.csv` | State × month road deaths, accidents and injuries, 2014–2024. Taken from the annexures of MoRTH's *Road Accidents in India* and reconciled to national totals. Archived with the MV Act fines study ([doi:10.5281/zenodo.24055110](https://doi.org/10.5281/zenodo.24055110)). |
| `data/state_month_weather_1991_2024.csv` | State × month weather for 35 units, 1991–2024: mean maximum and minimum temperature, days at or above 35 °C and 40 °C, total rainfall, rain days, heavy-rain days, and anomalies against the 1991–2020 normal. Population-weighted using Census 2011 districts. |
| `data/raw/census2011.csv` | Census 2011 district populations, used for the population weights. |
| `src/build_exposures.py` | Builds the state-month weather exposures from IMD gridded data (Pai et al. 2014; Srivastava et al. 2009). |
| `src/analysis_fe.py` | Fixed-effects analyses: joint heat and rain, lags and leads, cumulative effects, relative versus absolute heat, outcomes, seasons, bins. |
| `src/robust_ppml.py` | Poisson pseudo-maximum-likelihood robustness checks. |
| `src/bayes_hier.py` | Bayesian hierarchical negative-binomial model with state-varying lagged effects, displacement accounting and the projection contrast. |
| `src/kill_check.py`, `src/fig_heat.py` | Initial screening analysis; Figure 1. |
| `outputs/` | Results tables, posterior draws and log, figure. |
| `results/expected_key_results.json` | Values reported in the manuscript, checked by `verify.py`. |

## Licences

- **Code:** MIT.
- **Derived data:** CC BY 4.0.
- **IMD gridded data and MoRTH reports:** Government of India sources, cited and not redistributed here. The MoRTH reports are in the fines-study archive.

## Author

Siddalingaiah H S, Department of Community Medicine, Shridevi Institute of Medical Sciences and Research Hospital, Tumakuru, Karnataka, India. ORCID [0000-0002-4771-8285](https://orcid.org/0000-0002-4771-8285).
