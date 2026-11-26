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

# Preserve the complete original NAIS particle distribution
neg_p_nais_original = neg_p.copy()

# Preserve the complete original NAIS particle distribution
neg_p_nais_original = neg_p.copy()
neg_i_1 = neg_i.copy()

# Open and sort the DMPS dataset
dmps_df = pd.read_excel("/home/kivekia/Python/DMPS_BUCT_to_D_concentration.xlsx",
                        index_col=0,
                        parse_dates=True)

# DMPS timestamps are already Beijing local time
dmps_df.index = pd.to_datetime(dmps_df.index)

if dmps_df.index.tz is None:
    dmps_df.index = dmps_df.index.tz_localize("Asia/Shanghai")

dmps_df = dmps_df[
    ~dmps_df.index.duplicated(keep="first")]

dmps_df = dmps_df.sort_index()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Convert NAIS particle and ion distributions
nais_particles_dN = af.dndlogdp2dn(
    neg_p_nais_original.sort_index(axis=1)
)

nais_ions_dN = af.dndlogdp2dn(
    neg_i.sort_index(axis=1)
)

# Convert DMPS distribution
dmps_dN = af.dndlogdp2dn(
    dmps_df.sort_index(axis=1)
)

# --------------------------------------------------
# Interpolate DMPS in time to NAIS timestamps
# --------------------------------------------------

dmps_dN_10min = dmps_dN.reindex(
    nais_particles_dN.index.union(dmps_dN.index)
).sort_index()

dmps_dN_10min = dmps_dN_10min.interpolate(
    method="time",
    limit_area="inside"
)

dmps_dN_10min = dmps_dN_10min.reindex(
    nais_particles_dN.index
)

# --------------------------------------------------
# Interpolate DMPS onto the NAIS diameter grid
# --------------------------------------------------

# NAIS diameter grid (metres)
nais_dp = nais_particles_dN.columns.to_numpy(dtype=float)

# DMPS diameter grid (nm)
dmps_dp = dmps_dN_10min.columns.to_numpy(dtype=float)

# Start with NAIS particle distribution in dN
combined_dN = nais_particles_dN.copy()

# Select NAIS diameters between 10 and 40 nm
dmps_mask = (
    (nais_dp >= 10e-9) &
    (nais_dp <= 40e-9)
)

# Interpolate DMPS dN onto the NAIS diameter grid
for timestamp in nais_particles_dN.index:

    values = dmps_dN_10min.loc[timestamp].to_numpy(dtype=float)

    valid = (
        np.isfinite(values) &
        np.isfinite(dmps_dp) &
        (dmps_dp > 0)
    )

    if valid.sum() < 2:
        combined_dN.loc[timestamp, nais_dp[dmps_mask]] = np.nan
        continue

    combined_dN.loc[timestamp, nais_dp[dmps_mask]] = np.interp(
        np.log10(nais_dp[dmps_mask] * 1e9),
        np.log10(dmps_dp[valid]),
        values[valid],
        left=np.nan,
        right=np.nan
    )

# Use the combined particle distribution and converted NAIS ions
neg_p = combined_dN.copy()
neg_i = nais_ions_dN.copy()

# Remove negative concentrations
neg_i[neg_i < 0] = np.nan
neg_p[neg_p < 0] = np.nan

del dmps_dN_10min
del dmps_dN

# Open data for coagulation calculations
df = pd.read_excel("/home/kivekia/Python/BUCT DMPS 2018-2022/BUCT DMPS 2018-2022/DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

# DMPS timestamps are already Beijing local time
df.index = pd.to_datetime(df.index)

if df.index.tz is None:
    df.index = df.index.tz_localize("Asia/Shanghai")

df = df[
    ~df.index.duplicated(keep="first")]

df = df.sort_index()


df_10min = df.reindex(
    nais_particles_dN.index.union(df.index)
).sort_index()

df_10min = df_10min.interpolate(
    method="time",
    limit_area="inside"
)

df_10min = df_10min.reindex(
    nais_particles_dN.index
)

# Particle and ion concentrations on identical NAIS diameters
neg_p_charge = neg_p.copy()
neg_i_charge = neg_i.copy()

neg_i_charge[neg_i_charge < 0] = np.nan
neg_p_charge[neg_p_charge < 0] = np.nan

dp_nm = neg_i.columns.to_numpy(dtype=float)
dp_nm_2 = dp_nm * 1e9
df_10min.columns = df_10min.columns.astype(float) * 1e-9

coagN = af.calc_coags(df_10min, dp_nm)
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

N = af.calc_conc(neg_i_1, 0.0, 1.6e-9).squeeze()
if N.index.tz is None:
    N.index = N.index.tz_localize("Asia/Shanghai")
N = N.reindex(nais_particles_dN.index)

neg_p_charge = neg_p_charge.loc[:, neg_p_charge.columns >= 2.5e-9]
neg_i_charge = neg_i_charge.loc[:, neg_i_charge.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p_nais_original, 4e-9, 6e-9).squeeze()

threshold = 2000

mask = concentrations > threshold

neg_p_masked = neg_p_charge.copy()
neg_i_masked = neg_i_charge.copy()

neg_p_masked.loc[~mask, :] = np.nan
neg_i_masked.loc[~mask, :] = np.nan

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.2

f_meas = neg_i_masked / neg_p_masked

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

start = "2022-03-13 00:00:00"
end = "2022-03-20 23:59:59"
overcharge = overcharge.loc[start:end]

overcharge_30_min = overcharge.resample("1h").median()

S_small = overcharge_30_min.loc[
    :,
    (overcharge.columns >= 4e-9) &
    (overcharge.columns <= 6e-9)
]

S_small_median = S_small.median(axis=1)

S_large = overcharge_30_min.loc[
    :,
    (overcharge_30_min.columns >= 20e-9) &
    (overcharge_30_min.columns <= 30e-9)
]

S_large_median = S_large.median(axis=1)

concentrations_plotting = concentrations.copy()
concentrations_plotting.loc[~mask] = np.nan
concentrations_plotting = concentrations_plotting.loc[start:end]
concentrations_plotting = concentrations_plotting.resample("1h").median()

dk = xr.open_dataset("/home/kivekia/Python/wind_data.grib", engine="cfgrib")

# Calculate wind direction from ERA5 u10 and v10
u = dk["u10"].sel(
    latitude=40.0,
    longitude=116.3,
    method="nearest"
).to_pandas()

v = dk["v10"].sel(
    latitude=40.0,
    longitude=116.3,
    method="nearest"
).to_pandas()

wind_direction = (270 - np.degrees(np.arctan2(v, u))) % 360

wind = pd.DataFrame(
    {"wind_direction": wind_direction}
)

# Convert timestamps to Beijing local time
wind.index = pd.to_datetime(wind.index)

if wind.index.tz is None:
    wind.index = wind.index.tz_localize("UTC")

wind.index = wind.index.tz_convert("Asia/Shanghai")

# --------------------------------------------------
# Select wind directions between 270 and 360 degrees
# --------------------------------------------------

# Wind direction corresponding to the plotting timestamps
wind_plot = wind["wind_direction"].reindex(
    S_small_median.index,
    method="nearest",
    tolerance=pd.Timedelta("30min")
)

# Select timestamps with wind direction 270–360 degrees
wind_mask = (
    (wind_plot >= 270) &
    (wind_plot <= 360)
)

# Apply the same mask to all plotted variables
#S_small_median = S_small_median.loc[wind_mask]
#S_large_median = S_large_median.loc[wind_mask]

S_small_median = S_small_median.where(wind_mask)

S_large_median = S_large_median.where(wind_mask)

concentrations_plotting = concentrations_plotting.where(wind_mask)
# Print statistics
print("Total selected observations:", wind_mask.sum())

print(
    "Wind direction range:",
    wind_plot.loc[wind_mask].min(),
    "-",
    wind_plot.loc[wind_mask].max()
)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(
    S_large_median.index,
    S_large_median,
    label="S (20–30 nm)",
    linewidth=1.5,
    color = "green"
)

ax1.plot(
    S_small_median.index,
    S_small_median,
    label="S (4–6 nm)",
    linewidth=1.5,
    color = "blue"
)

ax1.set_xlabel("Time")
ax1.set_ylabel("S")

ax2 = ax1.twinx()

ax2.plot(
    concentrations_plotting.index,
    concentrations_plotting,
    label="N (4–6 nm)",
    linewidth=1.5,
    color = "red"
)

ax2.set_ylabel("Particle concentration (cm$^{-3}$)")

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig.autofmt_xdate()

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right"
)

ax1.set_title("Peking, 2022, week 11, with wind direction accounted for")
#ax1.set_ylim(-0.5, 9)
ax1.set_yscale("log")
ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("beijing_N_and_S_one_week_series_with_wind", bbox_inches='tight', dpi=300)