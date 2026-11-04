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

da_original=ds["neg_particles"]
da_original1=ds["neg_ions"]

da_original=da_original.copy()
da_original1=da_original1.copy()

da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

daa2=da_original1.dropna(dim="diameter", how="all")
daa=daa2.dropna(dim="time", how="any")

df=da.to_pandas()
dff=daa.to_pandas()

df.columns = df.columns.astype(float)
dff.columns = dff.columns.astype(float)

df = df.loc[:, df.columns <= 5e-9]
dff = dff.loc[:, dff.columns <= 5e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")
dff.index = dff.index.tz_localize("UTC").tz_convert("Europe/Rome")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ratio = dff / df

background = ratio.between_time('00:00', '04:00')

f_background = background.median()


overcharge = ratio.divide(
    f_background,
    axis=1
)

#df_30 = overcharge.resample('30min').median()

df_30 = overcharge


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

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    median_diurnal,
    ax,
    norm='linear',
    cmap='turbo'
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))

cbar.set_label('Charge ratio compared to background (Overcharge ratio)')


plt.savefig("milan_all_ratio5.png", bbox_inches='tight', dpi=300)