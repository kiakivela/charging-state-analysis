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

ds = xr.open_dataset("/home/kivekia/Python/beijing_all.nc")

ds = ds.sortby('time')

dmps_df = pd.read_excel("/home/kivekia/Python/DMPS_BUCT_to_D_concentration.xlsx",
                        index_col=0,
                        parse_dates=True)

dmps_df.index = (
    dmps_df.index
    .tz_localize("UTC")
    .tz_convert("Asia/Shanghai"))

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()

#negp_og2=negp_og.dropna(dim="diameter", how="all")
#negp=negp_og2.dropna(dim="time", how="all")

#negi_og2=negi_og.dropna(dim="diameter", how="all")
#negi=negi_og2.dropna(dim="time", how="all")

#neg_p=negp.to_pandas()
#neg_i=negi.to_pandas()

neg_p = negp_og.to_pandas()
neg_i = negi_og.to_pandas()

common_index = neg_p.index.intersection(neg_i.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i.loc[common_index]

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

# ============================================================
# INTERPOLATE DMPS TO NAIS 10-MINUTE TIME RESOLUTION
# ============================================================

# Remove duplicate timestamps if present
dmps_df = dmps_df[
    ~dmps_df.index.duplicated(keep="first")
]

# Sort time
dmps_df = dmps_df.sort_index()

# Interpolate hourly DMPS data to the NAIS timestamps
dmps_10min = dmps_df.reindex(
    neg_p.index.union(dmps_df.index)
).sort_index()

dmps_10min = dmps_10min.interpolate(
    method="time",
    limit_area="inside"
)

# Keep only the NAIS timestamps
dmps_10min = dmps_10min.reindex(neg_p.index)

# ============================================================
# COMBINE NAIS (<10 nm) AND DMPS (10–40 nm)
# ============================================================

# NAIS diameter columns are in meters
nais_diameters_nm = neg_p.columns.astype(float) * 1e9

# Select NAIS diameters below 10 nm
nais_below_10 = neg_p.loc[
    :, nais_diameters_nm < 10
].copy()

# DMPS columns are assumed to be in nm
dmps_diameters_nm = dmps_10min.columns.astype(float)

# Select DMPS diameters between 10 and 40 nm
dmps_10_40 = dmps_10min.loc[
    :, (dmps_diameters_nm >= 10) &
       (dmps_diameters_nm <= 40)
].copy()

# Convert DMPS diameter columns from nm to meters
dmps_10_40.columns = (
    dmps_10_40.columns.astype(float) * 1e-9
)

# Combine
combined = pd.concat(
    [nais_below_10, dmps_10_40],
    axis=1
)

# Sort by diameter
combined = combined.reindex(
    sorted(combined.columns),
    axis=1
)

neg_p = combined.copy()

neg_i[neg_i < 0] = np.nan
neg_p[neg_p < 0] = np.nan

dp_nm = np.array(neg_p.columns, dtype=float)
dp_nm_2 = dp_nm*1e9

df = pd.read_excel("/home/kivekia/Python/BUCT DMPS 2018-2022/BUCT DMPS 2018-2022/DMPS_BUCT_to_D.xlsx",
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

# ============================================================
# ALIGN NAIS IONS WITH COMBINED NAIS + DMPS PARTICLES
# ============================================================

# Original NAIS ion concentrations
neg_i_nais = neg_i_hourly.loc[common_index].copy()

# Combined particle concentrations
neg_p = neg_p_hourly.loc[common_index].copy()

# DMPS coagulation data
df = df_hourly.loc[common_index].copy()

# Source diameter grid: NAIS ions (meters)
ion_dp = neg_i_nais.columns.to_numpy(dtype=float)

# Target diameter grid: combined particles (meters)
particle_dp = neg_p.columns.to_numpy(dtype=float)

# Sort NAIS ion diameters
sort_idx = np.argsort(ion_dp)

ion_dp = ion_dp[sort_idx]

neg_i_nais = neg_i_nais.iloc[:, sort_idx]

# Convert diameters to logarithmic scale
log_ion_dp = np.log10(ion_dp)
log_particle_dp = np.log10(particle_dp)

# Initialize ion dataframe on the particle diameter grid
neg_i_aligned = pd.DataFrame(
    np.nan,
    index=neg_p.index,
    columns=particle_dp
)

# Keep original NAIS ion values below 10 nm
nais_mask = particle_dp < 10e-9

neg_i_aligned.loc[:, nais_mask] = (
    neg_i_nais.reindex(
        columns=particle_dp[nais_mask]
    )
)

# DMPS diameter range: 10–40 nm
dmps_mask = (
    (particle_dp >= 10e-9) &
    (particle_dp <= 40e-9)
)

# Interpolate NAIS ions onto DMPS diameters
for timestamp in neg_i_nais.index:

    ion_values = neg_i_nais.loc[timestamp].to_numpy(
        dtype=float
    )

    # Ignore missing ion measurements
    valid = np.isfinite(ion_values)

    if valid.sum() < 2:
        continue

    # Interpolate in log diameter
    interpolated = np.interp(
        log_particle_dp[dmps_mask],
        log_ion_dp[valid],
        ion_values[valid],
        left=np.nan,
        right=np.nan
    )

    neg_i_aligned.loc[
        timestamp,
        particle_dp[dmps_mask]
    ] = interpolated

# Use aligned ions from now on
neg_i = neg_i_aligned.copy()

df.columns = df.columns.astype(float) * 1e-9

coagN = af.calc_coags(df, dp_nm)
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

N = af.calc_conc(neg_i, 0.0, 1.6e-9).squeeze()

#neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
#neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p, 4e-9, 6e-9).squeeze()

threshold = 1315

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

#neg_p_masked = neg_p_masked.loc[:, neg_p_masked.columns <= 11e-9]
#neg_i_masked = neg_i_masked.loc[:, neg_i_masked.columns <= 11e-9]

#neg_p_masked = neg_p_masked.loc[:, neg_p_masked.columns >= 2.5e-9]
#neg_i_masked = neg_i_masked.loc[:, neg_i_masked.columns >= 2.5e-9]

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

diam_nm = np.array(neg_p.columns, dtype=float)

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