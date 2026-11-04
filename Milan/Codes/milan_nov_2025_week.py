import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/Milan/Nov 2025/*.nc"))
datasets = [xr.open_dataset(f) for f in files]

ds = xr.concat(datasets, dim='time')

ds = ds.sortby('time')

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

df.columns = df.columns.astype(float)

df = df.loc[:, df.columns >= 2e-9]


df_30=df

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")
#df_30 = df.resample('30min').median()

df_30['clocktime'] = df_30.index.strftime('%H:%M')

median_diurnal = df_30.groupby('clocktime').median()

times = sorted(median_diurnal.index)

median_diurnal = median_diurnal.loc[times]

#median_diurnal = median_diurnal.drop(
   # columns='time',
    #errors='ignore')

median_diurnal.index = pd.to_datetime(
    median_diurnal.index,
    format='%H:%M'
)

fig, ax = plt.subplots(figsize=(12,6))

ap.plot_aerosol_dist(median_diurnal, ax)

import matplotlib.dates as mdates

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.savefig("milan_nov_2025_week_particles.png", bbox_inches='tight', dpi=300)