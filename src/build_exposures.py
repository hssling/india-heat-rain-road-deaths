"""Population-weighted state-month weather exposures from IMD gridded daily data, 1991-2024.

Inputs : IMD gridded daily rainfall (0.25 deg) and max/min temperature (1 deg), via imdlib (data/imd/<var>/<yr>.grd)
         Census 2011 district polygons (districts_v2.geojson, year '2011_c') + populations (census2011.csv)
Method : each district takes the grid cells whose centres fall inside it (nearest cell to its centroid if none);
         district-day value = mean of those cells; state-month value = Census-2011-population-weighted mean of
         district-month values. Units match the road-death panel: Telangana's 10 districts are split out of Andhra
         Pradesh; J&K + Ladakh and DNH + DD are merged.
Metrics (per state-month): tmax_mean, tmin_mean, hot35 (days tmax >= 35 C), hot40 (days tmax >= 40 C),
         rain_total (mm), rain_days (>= 2.5 mm), heavy_days (>= 64.5 mm, IMD 'heavy'), plus anomalies vs the
         1991-2020 state x calendar-month normal.
Output : data/state_month_weather_1991_2024.csv
"""
import pathlib
import numpy as np, pandas as pd, geopandas as gpd
import imdlib as imd

P = pathlib.Path(__file__).resolve().parents[1]
A = P / "data/raw"   # census2011.csv (included) + districts_v2.geojson (see README)
IMD = P / "data/imd"
YEARS = range(1991, 2025)
TELANGANA = {"adilabad", "nizamabad", "karimnagar", "medak", "hyderabad", "rangareddy", "ranga reddy", "mahbubnagar",
             "mahabubnagar", "nalgonda", "warangal", "khammam"}
UNIT = {"Jammu and Kashmir": "J&K & Ladakh", "Ladakh": "J&K & Ladakh",
        "Dadra and Nagar Haveli and Daman and Diu": "DNH & DD", "Andaman and Nicobar Islands": "Andaman & Nicobar"}

# ---- districts with population ----
g = gpd.read_file(A / "districts_v2.geojson")
# the boundary file mixes vintages: Telangana exists only as its post-2016 districts, so a '2011_c' filter alone
# silently drops the whole state. Take 2011 districts plus Telangana's current ones; Census 2011 populations do not
# exist for the new Telangana districts, so within Telangana the (median-filled) weights are near-equal.
g = g[(g.year == "2011_c") | (g.st_nm == "Telangana")].copy()
c = pd.read_csv(A / "census2011.csv")
g["code"] = pd.to_numeric(g.dt_code, errors="coerce")
g = g.merge(c[["District code", "District name", "Population"]], left_on="code", right_on="District code", how="left")
miss = g.Population.isna()
byname = c.assign(k=c["District name"].str.lower().str.strip()).drop_duplicates("k").set_index("k").Population
g.loc[miss, "Population"] = g.loc[miss, "district"].str.lower().str.strip().map(byname)
print(f"districts {len(g)}, population matched {g.Population.notna().sum()} ({g.Population.sum():,.0f} people)")
g["Population"] = g.Population.fillna(g.Population.median())          # few unmatched: median, reported above
g["unit"] = g.st_nm.map(lambda s: UNIT.get(s, s))
g.loc[(g.st_nm == "Andhra Pradesh") & g.district.str.lower().str.strip().isin(TELANGANA), "unit"] = "Telangana"
print("Telangana districts:", (g.unit == "Telangana").sum())


def cell_index(lat, lon):
    """district -> list of (i, j) grid indices: cells with centres inside, else nearest to centroid."""
    LON, LAT = np.meshgrid(lon, lat)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs="EPSG:4326")
    pts["ij"] = list(zip(*np.unravel_index(np.arange(LAT.size), LAT.shape)))
    j = gpd.sjoin(pts, g[["geometry"]].reset_index(names="d"), predicate="within")
    idx = j.groupby("d").ij.apply(list).to_dict()
    for d, row in g.iterrows():
        if d not in idx:
            cen = row.geometry.representative_point()
            idx[d] = [(int(np.abs(lat - cen.y).argmin()), int(np.abs(lon - cen.x).argmin()))]
    return idx


def district_daily(var, yr, idx_cache):
    ds = imd.open_data(var, yr, yr, "yearwise", str(IMD)).get_xarray()
    arr = ds[var].values.astype("float64")                             # time x lat x lon
    arr[arr <= -999] = np.nan
    key = (len(ds.lat), len(ds.lon))
    if key not in idx_cache:
        idx_cache[key] = cell_index(ds.lat.values, ds.lon.values)
    idx = idx_cache[key]
    out = np.full((arr.shape[0], len(g)), np.nan)
    for d, cells in idx.items():
        ii, jj = zip(*cells)
        out[:, d] = np.nanmean(arr[:, ii, jj], axis=1)
    return pd.DataFrame(out, index=pd.to_datetime(ds.time.values))


cache, frames = {}, []
for yr in YEARS:
    tx, tn, rf = (district_daily(v, yr, cache) for v in ("tmax", "tmin", "rain"))
    for d in range(len(g)):                                            # empty district series (e.g. ocean cells)
        for df in (tx, tn, rf):
            if df[d].isna().all(): df[d] = np.nan
    m = tx.index.to_period("M")
    mon = pd.concat({
        "tmax_mean": tx.groupby(m).mean(), "tmin_mean": tn.groupby(m).mean(),
        "hot35": (tx >= 35).groupby(m).sum(), "hot40": (tx >= 40).groupby(m).sum(),
        "rain_total": rf.groupby(m).sum(min_count=1), "rain_days": (rf >= 2.5).groupby(m).sum(),
        "heavy_days": (rf >= 64.5).groupby(m).sum()}, axis=1)
    frames.append(mon)
    print(yr, "done", flush=True)
D_ = pd.concat(frames)                                                 # month x (metric, district)
w = g.Population.values
rows = []
for unit, idxs in g.groupby("unit").groups.items():
    idxs = list(idxs)
    for metric in D_.columns.levels[0]:
        X = D_[metric][idxs].values                                    # month x districts
        ww = np.where(np.isnan(X), 0, w[idxs])
        val = np.nansum(X * ww, 1) / ww.sum(1)
        rows.append(pd.DataFrame({"state": unit, "date": D_.index.to_timestamp(), "metric": metric, "value": val}))
out = pd.concat(rows).pivot_table(index=["state", "date"], columns="metric", values="value").reset_index()
out["cal"] = out.date.dt.month
norm = out[(out.date.dt.year >= 1991) & (out.date.dt.year <= 2020)].groupby(["state", "cal"]).mean(numeric_only=True)
for metric in ("tmax_mean", "tmin_mean", "hot35", "hot40", "rain_total", "rain_days", "heavy_days"):
    out[f"{metric}_anom"] = out[metric] - out.set_index(["state", "cal"]).index.map(norm[metric])
out.drop(columns="cal").to_csv(P / "data/state_month_weather_1991_2024.csv", index=False)
print("saved", out.shape, "| states:", out.state.nunique())
print(out[out.state.isin(["Rajasthan", "Kerala"]) & (out.date.dt.year == 2024)].groupby("state")[["tmax_mean", "hot40", "rain_total"]].agg(["mean", "max"]).round(1))
