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

ds = xr.open_dataset("/home/kivekia/Python/beijing_all.nc")

ds = ds.sortby('time')

# ============================================================
# READ DMPS DATA
# ============================================================

dmps_df = pd.read_excel("/home/kivekia/Python/DMPS_BUCT_to_D_concentration.xlsx",
                        index_col=0,
                        parse_dates=True)

dmps_df.index = (
    dmps_df.index
    .tz_localize("UTC")
    .tz_convert("Asia/Shanghai")
)

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

# Check diameter range
print(
    "Combined diameter range (nm):",
    combined.columns.min() * 1e9,
    combined.columns.max() * 1e9
)

# ============================================================
# DAILY MEDIAN SIZE DISTRIBUTION (01:00–03:00)
# ============================================================

night_mask = (
    (combined.index.hour >= 1) &
    (combined.index.hour < 3)
)

combined_night = combined.loc[night_mask]

daily_median = combined_night.groupby(
    combined_night.index.date
).median()

daily_median.index = pd.to_datetime(
    daily_median.index
)

daily_median.index.name = "Date"

daily_median = daily_median.sort_index()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Plot daily median size distributions
fig, ax = plt.subplots(figsize=(12, 6))

im, cbar = ap.plot_aerosol_dist(
    daily_median,
    ax,
    norm="linear",
)

ax.set_xlabel("Date")
ax.set_ylabel("Particle diameter (nm)")
ax.set_title(
    "Daily median negative particle size distribution "
    "(01:00–03:00 Beijing time)"
)

# Format x-axis as dates
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m")
)
ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=3)
)

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("testing_DMPS.png", dpi=300)

fig, ax = plt.subplots(figsize=(12, 6))

from matplotlib.colors import LogNorm

im = ax.pcolormesh(
    daily_median.index,
    daily_median.columns.to_numpy() * 1e9,
    daily_median.T,
    shading="auto",
    cmap="coolwarm",
    norm=LogNorm(vmin=100, vmax=1e5)
)

fig.colorbar(im, ax=ax, label="Particle concentration")

ax.set_yscale("log")
ax.set_ylabel("Particle diameter (nm)")
ax.set_xlabel("Date")


ax.set_title("Combined NAIS + DMPS distribution")

plt.tight_layout()
plt.savefig("combined_night_median_distribution.png", dpi=300)