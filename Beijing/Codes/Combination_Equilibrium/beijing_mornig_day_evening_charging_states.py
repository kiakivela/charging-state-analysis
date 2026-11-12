import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import aerosol.lab as al

import matplotlib
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = matplotlib.colormaps.get_cmap

# Open and sort the NAIS dataset
ds = xr.open_dataset("/home/kivekia/Python/beijing_all.nc")

ds = ds.sortby('time')

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()

neg_p = negp_og.to_pandas()
neg_i = negi_og.to_pandas()

common_index = neg_p.index.intersection(neg_i.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i.loc[common_index]

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

# Preserve the complete original NAIS particle distribution
neg_p_nais_original = neg_p.copy()

# Open and sort the DMPS dataset
dmps_df = pd.read_excel("/home/kivekia/Python/DMPS_BUCT_to_D_concentration.xlsx",
                        index_col=0,
                        parse_dates=True)

dmps_df.index = (
    dmps_df.index
    .tz_localize("UTC")
    .tz_convert("Asia/Shanghai"))

dmps_df = dmps_df[
    ~dmps_df.index.duplicated(keep="first")]

dmps_df = dmps_df.sort_index()

# Interpolate DMPS data to 10-minute intervals (same as NAIS) and reindex to match NAIS timestamps
dmps_10min = dmps_df.reindex(
    neg_p.index.union(dmps_df.index)
).sort_index()

dmps_10min = dmps_10min.interpolate(
    method="time",
    limit_area="inside"
)

dmps_10min = dmps_10min.reindex(neg_p.index)

# Original NAIS diameter grid (metres)
nais_dp = neg_p.columns.to_numpy(dtype=float)

# DMPS diameter grid (nm)
dmps_dp = dmps_10min.columns.to_numpy(dtype=float)

# Start with original NAIS particle distribution
combined = neg_p_nais_original.copy()

# Select NAIS diameters between 10 and 40 nm
dmps_mask = (
    (nais_dp >= 10e-9) &
    (nais_dp <= 40e-9)
)

# Interpolate DMPS concentrations onto NAIS diameter grid
for timestamp in neg_p.index:

    values = dmps_10min.loc[timestamp].to_numpy(dtype=float)

    valid = np.isfinite(values)

    if valid.sum() < 2:
        combined.loc[timestamp, nais_dp[dmps_mask]] = np.nan
        continue

    combined.loc[timestamp, nais_dp[dmps_mask]] = np.interp(
        np.log10(nais_dp[dmps_mask] * 1e9),
        np.log10(dmps_dp[valid]),
        values[valid],
        left=np.nan,
        right=np.nan
    )

neg_p = combined.copy()

neg_i[neg_i < 0] = np.nan
neg_p[neg_p < 0] = np.nan

# Open data for coagulation calculations
df = pd.read_excel("/home/kivekia/Python/BUCT DMPS 2018-2022/BUCT DMPS 2018-2022/DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

# Change the index to 1-hour intervals
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

neg_p_nais_hourly = (
    neg_p_nais_original
    .resample("1h")
    .median()
    .dropna(how="all")
)

common_index = (
    neg_i_hourly.index
    .intersection(neg_p_hourly.index)
    .intersection(neg_p_nais_hourly.index)
    .intersection(df_hourly.index)
)

neg_i_nais = neg_i_hourly.loc[common_index].copy()
neg_p_nais = neg_p_hourly.loc[common_index].copy()
df = df_hourly.loc[common_index].copy()

# Particle and ion concentrations on identical NAIS diameters
neg_p_charge = neg_p_nais.copy()
neg_i_charge = neg_i_nais.copy()

neg_i_charge[neg_i_charge < 0] = np.nan
neg_p_charge[neg_p_charge < 0] = np.nan

dp_nm = neg_i.columns.to_numpy(dtype=float)
dp_nm_2 = dp_nm * 1e9
df.columns = df.columns.astype(float) * 1e-9

coagN = af.calc_coags(df, dp_nm)
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

N = af.calc_conc(neg_i_charge, 0.0, 1.6e-9).squeeze()

neg_p_charge = neg_p_charge.loc[:, neg_p_charge.columns >= 2.5e-9]
neg_i_charge = neg_i_charge.loc[:, neg_i_charge.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p_charge, 4e-9, 6e-9).squeeze()

threshold = 1315

mask = concentrations > threshold

neg_p_masked = neg_p_charge.copy()
neg_i_masked = neg_i_charge.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.2

f_meas = neg_i_masked / neg_p_masked

print(f_meas.isna().mean() * 100)

dp_nm = neg_p_charge.columns.to_numpy(dtype=float)

f_eq_1 = pd.Series(
    {
        diameter: al.eq_charge_frac(diameter, -1)
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
    index=neg_p_charge.index,
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
    .between_time("06:00", "09:00")
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

diam_nm = np.array(neg_i.columns, dtype=float)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
line1 = ax1.plot(
    diam_nm,
    morning,
    label=r"Median charge fraction, klo 6-9",
)

line4 = ax1.plot(
    diam_nm,
    day,
    label=r"Median charge fraction, klo 11-13",
)

line5 = ax1.plot(
    diam_nm,
    evening,
    label=r"Median charge fraction, klo 17-19",
)

ax1.set_xlabel("Diameter (m)")
ax1.set_ylabel("Charge fraction")
ax1.set_xscale("log")
ax1.set_yscale("log")
#ax1.set_xlim(2e-9, 20e-9)
#ax1.set_ylim(0, 0.2)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.savefig("beijing_morning_day_evening_charging_states.png", dpi=300)