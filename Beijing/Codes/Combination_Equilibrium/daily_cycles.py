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

#dmps_df.index = (
#    dmps_df.index
#    .tz_localize("UTC")
#    .tz_convert("Asia/Shanghai"))

# DMPS timestamps are already Beijing local time
dmps_df.index = pd.to_datetime(dmps_df.index)

if dmps_df.index.tz is None:
    dmps_df.index = dmps_df.index.tz_localize("Asia/Shanghai")

dmps_df = dmps_df[
    ~dmps_df.index.duplicated(keep="first")]

dmps_df = dmps_df.sort_index()

import matplotlib.dates as mdates

# Select the day in Beijing local time
day = "2020-01-07"

start = pd.Timestamp(day, tz="Asia/Shanghai")
end = start + pd.Timedelta(days=1)

# Select the same day from both instruments
nais_day = neg_p_nais_original.loc[
    (neg_p_nais_original.index >= start) &
    (neg_p_nais_original.index < end)
].copy()

dmps_day = dmps_df.loc[
    (dmps_df.index >= start) &
    (dmps_df.index < end)
].copy()

# Convert NAIS diameter from metres to nm
nais_day.columns = nais_day.columns.astype(float) * 1e9

# Convert DMPS diameter from metres to nm if its columns are in metres.
# If the DMPS columns are already in nm, do not multiply by 1e9.
dmps_day.columns = dmps_day.columns.astype(float)

# Keep only the 10–15 nm size range
nais_day = nais_day.loc[
    :, (nais_day.columns >= 10) & (nais_day.columns <= 15)
]

dmps_day = dmps_day.loc[
    :, (dmps_day.columns >= 10) & (dmps_day.columns <= 15)
]

if nais_day.empty:
    raise ValueError(f"No NAIS measurements found for {day}")

if dmps_day.empty:
    raise ValueError(f"No DMPS measurements found for {day}")

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 9),
    sharex=True
)

# NAIS
im1, cbar1 = ap.plot_aerosol_dist(
    nais_day,
    axes[0],
    norm='linear'
)

axes[0].set_title(f"NAIS – {day}")
axes[0].set_ylabel("Diameter (nm)")

# DMPS
im2, cbar2 = ap.plot_aerosol_dist(
    dmps_day,
    axes[1],
    norm='linear'
)

axes[1].set_title(f"DMPS – {day}")
axes[1].set_ylabel("Diameter (nm)")
axes[1].set_xlabel("Time")

# Match the colour scales
#im1.set_clim(1, 250)
#im2.set_clim(1, 250)

# Format time axis
axes[1].xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M', tz=dmps_day.index.tz)
)

axes[1].xaxis.set_major_locator(
    mdates.HourLocator(interval=2)
)

# Colourbars
cbar1.set_label("dN/dlogDp")
cbar2.set_label("dN/dlogDp")

plt.tight_layout()

plt.savefig(
    f"NAIS_DMPS_daily_cycle_{day}.png",
    bbox_inches="tight",
    dpi=300
)