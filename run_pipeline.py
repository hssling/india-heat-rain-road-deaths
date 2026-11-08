"""Re-run the analysis: python run_pipeline.py [--bayes] [--build]
  default : fixed-effects and Poisson-PML analyses (a few minutes), figure, then verify.py
  --bayes : refit the Bayesian hierarchical model (NumPyro, ~2 min; pip install -r requirements-bayes.txt)
  --prior-sensitivity : refit it under the main, wide and vague priors (Table S5; ~6 min; same requirements)
  --build : rebuild data/state_month_weather_1991_2024.csv from IMD gridded data (downloads ~0.9 GB with imdlib;
            needs data/raw/districts_v2.geojson, see README)"""
import pathlib, subprocess, sys
R = pathlib.Path(__file__).resolve().parent
run = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=R, check=True)
if "--build" in sys.argv:
    import imdlib as imd
    for var in ("tmax", "tmin", "rain"): imd.get_data(var, 1991, 2024, fn_format="yearwise", file_dir=str(R / "data/imd"))
    run("src/build_exposures.py")
run("src/analysis_fe.py"); run("src/robust_ppml.py")
if "--bayes" in sys.argv: run("src/bayes_hier.py")
if "--prior-sensitivity" in sys.argv: run("src/bayes_prior_sensitivity.py")
run("src/fig_heat.py"); run("verify.py")
