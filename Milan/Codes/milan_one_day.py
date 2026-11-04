import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/milan/NAIS_20240712.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 11e-9]
df.loc[:, df.columns <= 2e-9]= np.nan

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(8,6))
im, cbar = ap.plot_aerosol_dist(df, ax, norm="log", clim=[1e3, 1e6])
plt.tight_layout

cbar.set_ticks([1e3, 1e4, 1e5])
cbar.set_ticklabels(["10³", "10⁴", "10⁵"])

plt.savefig("milan_20240712.png", bbox_inches='tight', dpi=300)