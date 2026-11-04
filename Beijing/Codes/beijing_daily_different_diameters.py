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

neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p, 2.5e-9, 5e-9).squeeze()

p = np.percentile(concentrations, 40)
print(p)

threshold = 1000

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

neg_p_masked = neg_p_masked.loc[:, neg_p_masked.columns <= 11e-9]
neg_i_masked = neg_i_masked.loc[:, neg_i_masked.columns <= 11e-9]

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

overcharge = f_meas.divide(
    f_eq,
    axis=1
)

start = "2020-10-01"
end = "2023-12-31 23:59:59"

overcharge = overcharge.loc[start:end]

df_30 = overcharge

df_30 = df_30.dropna(how="all")

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

first = median_diurnal.loc[:, median_diurnal.columns >= 3e-9]
first = median_diurnal.loc[:, median_diurnal.columns <= 4e-9]

first = first.median(axis=1)

second = median_diurnal.loc[:, median_diurnal.columns >= 4e-9]
second = median_diurnal.loc[:, median_diurnal.columns <= 5e-9]

second = second.median(axis=1)

third = median_diurnal.loc[:, median_diurnal.columns >= 5e-9]
third = median_diurnal.loc[:, median_diurnal.columns <= 6e-9]

third = third.median(axis=1)

fourth = median_diurnal.loc[:, median_diurnal.columns >= 6e-9]
fourth = median_diurnal.loc[:, median_diurnal.columns <= 7e-9]

fourth = fourth.median(axis=1)

fifth = median_diurnal.loc[:, median_diurnal.columns >= 7e-9]
fifth = median_diurnal.loc[:, median_diurnal.columns <= 8e-9]

fifth = fifth.median(axis=1)

sixth = median_diurnal.loc[:, median_diurnal.columns >= 8e-9]
sixth = median_diurnal.loc[:, median_diurnal.columns <= 9e-9]

sixth = sixth.median(axis=1)

seventh = median_diurnal.loc[:, median_diurnal.columns >= 9e-9]
seventh = median_diurnal.loc[:, median_diurnal.columns <= 10e-9]

seventh = seventh.median(axis=1)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
line1 = ax1.plot(
    median_diurnal.index,
    first,
    label=r"3-4 nm",
)

line2 = ax1.plot(
    median_diurnal.index,
    second,
    label=r"4-5 nm",
)

line3 = ax1.plot(
    median_diurnal.index,
    third,
    label=r"5-6 nm",
)

line4 = ax1.plot(
    median_diurnal.index,
    fourth,
    label=r"6-7 nm",
)

line5 = ax1.plot(
    median_diurnal.index,
    fifth,
    label=r"7-8 nm",
)

line6 = ax1.plot(
    median_diurnal.index,
    sixth,
    label=r"8-9 nm",
)

line7 = ax1.plot(
    median_diurnal.index,
    seventh,
    label=r"9-10 nm",
)

ax1.set_xlabel("Time")
ax1.set_ylabel("S")
ax1.set_yscale("log")
#ax1.set_xlim(2e-9, 20e-9)
#ax1.set_ylim(0, 0.2)

ax1.xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M")
)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()