import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/milan/*.nc"))
datasets = [xr.open_dataset(f) for f in files]

ds = xr.concat(datasets, dim='time')

ds = ds.sortby('time')

da_original=ds["neg_ions"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")

median_diurnal = df.groupby(df.index.time).median()

median_diurnal.index = pd.date_range(
    start='2000-01-01',
    periods=len(median_diurnal),
    freq='30min'
)

fig, ax = plt.subplots(figsize=(12,6))

ap.plot_aerosol_dist(median_diurnal, ax)

import matplotlib.dates as mdates

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.savefig("milanall.png", bbox_inches='tight', dpi=300)