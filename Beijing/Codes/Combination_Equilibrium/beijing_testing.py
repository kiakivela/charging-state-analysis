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

threshold = 1315

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

df_30 = overcharge

df_30 = df_30.dropna(how="all")

df_30["clocktime"] = df_30.index.strftime('%H:%M')

median_diurnal = (
    df_30
    .groupby("clocktime")
    .median(numeric_only=True)
)

morning = (
    overcharge
    .between_time("06:00", "08:00")
    .median(axis=0)
)

day = (
    overcharge
    .between_time("11:00", "13:00")
    .median(axis=0)
)

evening = (
    overcharge
    .between_time("17:00", "19:00")
    .median(axis=0)
)

d = neg_p.columns*1e9
diam_nm = np.array(d, dtype=float)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
line1 = ax1.plot(
    diam_nm,
    morning,
    label=r"Median S, klo 6-9",
)

line4 = ax1.plot(
    diam_nm,
    day,
    label=r"Median S, klo 11-13",
)

line5 = ax1.plot(
    diam_nm,
    evening,
    label=r"Median S, klo 17-19",
)

ax1.set_xlabel("Diameter (nm)")
ax1.set_ylabel("S")
ax1.set_xscale("log")
ax1.set_yscale("log")
#ax1.set_xlim(4, 50)
#ax1.set_ylim(0, 1)

ax1.set_xticks([4, 6, 10, 20, 30, 40],labels=[4, 6, 10, 20, 30, 40])
ax1.set_yticks([0.4, 0.6, 1, 2, 3, 4],labels=[0.4, 0.6, 1, 2, 3, 4])

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()

# Calculate morning/day ratio
morning_day_ratio = morning / day

diam_nm = np.array(neg_p.columns, dtype=float) * 1e9

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    diam_nm,
    morning_day_ratio,
    label="Morning / Daytime",
    linewidth=2
)

ax.set_xlabel("Diameter (nm)")
ax.set_ylabel("Morning / Daytime median overcharge ratio")

ax.set_xscale("log")
ax.set_xlim(4, 50)

ax.set_xticks(
    [4, 6, 10, 20, 30, 40],
    labels=[4, 6, 10, 20, 30, 40]
)

ax.axhline(1, color="gray", linestyle="--", linewidth=1)
ax.grid(True, which="both", alpha=0.25)
ax.legend()

plt.tight_layout()
plt.show()