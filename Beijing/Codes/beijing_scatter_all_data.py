import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap

import matplotlib
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = matplotlib.colormaps.get_cmap

ds = xr.open_dataset(r"C:\Users\kivekia\Downloads\Python\beijing_all.nc")

ds = ds.sortby('time')

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()

negp_og2=negp_og.dropna(dim="diameter", how="all")
negp=negp_og2.dropna(dim="time", how="all")

negi_og2=negi_og.dropna(dim="diameter", how="all")
negi=negi_og2.dropna(dim="time", how="all")

neg_p=negp.to_pandas()
neg_i=negi.to_pandas()

common_index = neg_p.index.intersection(neg_i.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i.loc[common_index]

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

neg_i[neg_i < 0] = np.nan
neg_p[neg_p < 0] = np.nan

dp_nm = np.array(neg_p.columns, dtype=float)
dp_nm_2 = dp_nm*1e9

df = pd.read_excel(r"C:\Users\kivekia\Downloads\Python\BUCT DMPS 2018-2022\BUCT DMPS 2018-2022\DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

neg_i_raw = neg_i.copy()
neg_p_raw = neg_p.copy()

neg_i_hourly = neg_i_raw.resample("1h").median()
neg_p_hourly = neg_p_raw.resample("1h").median()

neg_i_hourly = neg_i_hourly.dropna(how="all")
neg_p_hourly = neg_p_hourly.dropna(how="all")

df_hourly = df.copy()
df_hourly.index = df_hourly.index.floor("1h")
df_hourly = df_hourly.loc[
    ~df_hourly.index.duplicated(keep="first")
]

common_index = (
    neg_i_hourly.index
    .intersection(neg_p_hourly.index)
    .intersection(df_hourly.index)
)

neg_i = neg_i_hourly.loc[common_index].copy()
neg_p = neg_p_hourly.loc[common_index].copy()
df = df_hourly.loc[common_index].copy()

df.columns = df.columns.astype(float) * 1e-9

coagN = af.calc_coags(df, dp_nm)
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)


N = af.calc_conc(neg_i, 0.0, 1.6e-9).squeeze()

#neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
#neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p, 2.5e-9, 5e-9).squeeze()

#print(np.percentile(concentrations, 40))

threshold = 1000

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

#neg_p_masked = neg_p_masked.loc[:, neg_p_masked.columns <= 11e-9]
#neg_i_masked = neg_i_masked.loc[:, neg_i_masked.columns <= 11e-9]

neg_p_masked = neg_p_masked.loc[:, neg_p_masked.columns >= 2.5e-9]
neg_i_masked = neg_i_masked.loc[:, neg_i_masked.columns >= 2.5e-9]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.2
dp_nm = np.array(neg_p.columns, dtype=float)

f_meas = neg_i / neg_p

f_eq_1 = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in neg_p.columns
    }
)

f_eq = pd.DataFrame(
    index=neg_p.index,
    columns=dp_nm,
    dtype=float
)

for diameter in dp_nm:

    A = a0 * (diameter / d0)**l

    f_eq[diameter] = (
        A * N
    ) / (
        a * N
        + coagS[diameter]
        - coagN[diameter]
    )

#- 2 * A * N

overcharge_1 = f_meas/f_eq_1
overcharge = f_meas.divide(
    f_eq,
    axis=1
)

start = "2020-10-01 00:00:00"
end = "2022-06-29 23:59:59"

overcharge_1 = overcharge_1.loc[start:end]

S_small = overcharge.loc[
    :,
    (overcharge.columns >= 2.5e-9) &
    (overcharge.columns <= 4e-9)
]

S_small_median = S_small.median(axis=1)

S_large = overcharge_1.loc[
    :,
    (overcharge_1.columns >= 3e-9) &
    (overcharge_1.columns <= 4e-9)
]

S_large_median = S_large.median(axis=1)

concentrations_plotting = af.calc_conc(neg_p, 3e-9, 4e-9).squeeze()
#concentrations_plotting = concentrations_plotting.loc[start:end]

if isinstance(concentrations_plotting, pd.DataFrame):
    concentrations_plotting = concentrations_plotting.iloc[:, 0]

S_large_period = S_large_median.loc[start:end]
concentration_period = concentrations_plotting.loc[start:end]

scatter_data = pd.concat(
    [
        concentration_period.rename("concentration"),
        S_large_period.rename("S_large")
    ],
    axis=1
).dropna()

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    scatter_data["concentration"],
    scatter_data["S_large"],
    s=5,          # smaller dots
    alpha=0.4
)

time_mask = (
    (scatter_data.index.hour >= 4) &
    (scatter_data.index.hour < 8)
)

morning_data = scatter_data.loc[time_mask]

# Plot 04:00–08:00 in red
ax.scatter(
    morning_data["concentration"],
    morning_data["S_large"],
   s=5,
    alpha=0.8,
    color="red",
    label="04:00–08:00"
)

ax.set_xscale("log")
ax.set_yscale("log")

#ax.set_ylim(0.01, 10)

ax.set_xlabel("Particle concentration (3–4 nm, cm$^{-3}$)")
ax.set_ylabel("S 3–4 nm)")

plt.tight_layout()
plt.show()

from scipy.stats import pearsonr

# Select the two variables
corr_data = morning_data[["concentration", "S_large"]].copy()

# Remove NaN and infinite values
corr_data = corr_data.replace([np.inf, -np.inf], np.nan).dropna()

corr_data = corr_data[
    (corr_data["concentration"] > 0) &
    (corr_data["S_large"] > 0)
]

# Check how many points remain
print("Number of valid points:", len(corr_data))

# Calculate Pearson correlation
r, p = pearsonr(
    np.log10(corr_data["concentration"]),
    np.log10(corr_data["S_large"])
)

print(f"Pearson correlation coefficient (r) = {r:.4f}")
print(f"p-value = {p:.4e}")
