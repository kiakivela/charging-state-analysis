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
# DAILY MEDIAN SIZE DISTRIBUTION (01:00–03:00)
# ============================================================

# Select measurements between 01:00 and 03:00 Beijing time
night_mask = (
    (neg_p.index.hour >= 1) &
    (neg_p.index.hour < 3)
)

neg_p_night = neg_p.loc[night_mask]

# Calculate one median value per day for each particle diameter
daily_median = neg_p_night.groupby(
    neg_p_night.index.date
).median()

# Convert the index to datetime
daily_median.index = pd.to_datetime(daily_median.index)
daily_median.index.name = "Date"

# Sort by date
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
plt.show()