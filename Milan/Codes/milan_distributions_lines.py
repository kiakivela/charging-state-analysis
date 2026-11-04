import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
import math
import matplotlib.dates as mdates

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/milan/NAIS_20260330.nc")

#files = sorted(glob.glob("/home/kivekia/Summer Work 2026/Milan/Apr 2026/*.nc"))
#datasets = [xr.open_dataset(f) for f in files]
#ds = xr.concat(datasets, dim='time')
#ds = ds.sortby('time')

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="all")

df=da.to_pandas()
#print (df)

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")

#print(df)

concentrations = af.calc_conc(df, 2.5e-9, 5e-9)

threshold_percentile = np.percentile(concentrations, 90)

threshold = 5250

concentrations = concentrations.squeeze() 

mask = concentrations > threshold

event_times_horizontal = concentrations.index[mask]

#df = df.loc[:, df.columns <= 11e-9]
#df.loc[:, df.columns <= 2e-9] = np.nan

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    df,
    ax,
    norm="log",
    clim=[1e3, 1e6]
)


for t in event_times_horizontal:
    ax.hlines(
        y=np.log10(4e-9),
        xmin=t - pd.Timedelta("10min"),
        xmax=t + pd.Timedelta("10min"),
        color='black',
        linewidth=2,
        zorder=10
    )

#for t in event_times: 
# ax.axvline(t, color='black', linewidth=1)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

cbar.set_ticks([1e3, 1e4, 1e5])
cbar.set_ticklabels(["10³", "10⁴", "10⁵"])

plt.savefig("milan_extreme_lines_20260330_set_threshold_conc.png", bbox_inches="tight", dpi=300)