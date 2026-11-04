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

concentrations = af.calc_conc_interp(neg_p, 4e-9, 6e-9).squeeze()

#print(np.percentile(concentrations, 40))

threshold = 1295

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.2
dp_nm = np.array(neg_p.columns, dtype=float)

f_meas = neg_i_masked / neg_p_masked

f_eq_1 = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in neg_p.columns
    }
)

f_eq_2 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)/a
        for diameter in dp_nm
    }
)

f_eq_3 = pd.DataFrame(
    index=neg_p.index,
    columns=dp_nm,
    dtype=float
)

for diameter in dp_nm:

    A = a0 * (diameter / d0)**l

    f_eq_3[diameter] = (
        A * N
    ) / (
        a * N
        + coagS[diameter]
        - coagN[diameter]
    )

f_eq = f_eq_1 - f_eq_2 + f_eq_3

overcharge = f_meas.divide(
    f_eq,
    axis=1
)
start = "2020-10-01 00:00:00"
end = "2022-06-29 23:59:59"

overcharge = overcharge.loc[start:end]

overcharge = overcharge[
    (overcharge.index.hour >= 6) &
    (overcharge.index.hour < 23)
]

S_small = overcharge.loc[
    :,
    (overcharge.columns >= 4e-9) &
    (overcharge.columns <= 6e-9)
]

S_large = overcharge.loc[
    :,
    (overcharge.columns >= 15e-9) &
    (overcharge.columns <= 25e-9)
]

S_small_median = S_small.median(axis=1)
S_large_median = S_large.median(axis=1)

ratio = S_small_median / S_large_median


grouped = ratio.groupby(
    ratio.index.strftime('%H:%M')
)

p25 = grouped.quantile(0.25)
p50 = grouped.quantile(0.50)
p75 = grouped.quantile(0.75)

# convert to datetime
for s in [p25, p50, p75]:
    s.index = pd.to_datetime(s.index, format='%H:%M')

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(p25.index, p25.values, label='25th percentile')
ax.plot(p50.index, p50.values, label='50th percentile')
ax.plot(p75.index, p75.values, label='75th percentile')

ax.set_xlabel("Time of day")
ax.set_ylabel("Ratio")
ax.set_yscale("log")

#ax.set_ylim(0, 7)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

df_30 = overcharge

df_30["clocktime"] = df_30.index.strftime('%H:%M')

median_diurnal = (
    df_30
    .groupby("clocktime")
    .median(numeric_only=True)
)

times = sorted(median_diurnal.index)

median_diurnal = median_diurnal.loc[times]


median_diurnal.index = pd.to_datetime(
    median_diurnal.index,
    format='%H:%M')

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    median_diurnal,
    ax,
    norm='linear',
    #clim=[0.28, 0.60]
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))


#cbar.set_ticks([0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
#cbar.set_ticklabels(["0.30", "0.35", "0.40", "0.45", "0.50", "0.55"])

cbar.set_label('Overcharge ratio')
plt.show()

for d in [10e-9, 15e-9, 20e-9, 30e-9, 40e-9]:
    diameter = neg_p.columns[
        np.argmin(np.abs(neg_p.columns - d))
    ]

    print(f"\nDiameter: {diameter * 1e9:.1f} nm")
    print("Measured fraction:",
          f_meas[diameter].median())
    print("Equilibrium fraction:",
          f_eq[diameter].median())
    print("Charging state:",
          overcharge[diameter].median())