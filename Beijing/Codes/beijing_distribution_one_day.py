import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import matplotlib.dates as mdates
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = plt.get_cmap


ds = xr.open_dataset(r"C:\Users\kivekia\Downloads\Python\Summer Work 2026\beijing\NAIS_20190121.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 11e-9]
#df = df.loc[:, df.columns >= 2.5e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

coagN = af.calc_coags(df, df.columns[0:39])

fig, ax = plt.subplots(figsize=(12,6))
im, cbar = ap.plot_aerosol_dist(df, ax, norm="log")
plt.tight_layout

#clim=[1e3, 1e6]

#cbar.set_ticks([1e3, 1e4, 1e5])
#cbar.set_ticklabels(["10³", "10⁴", "10⁵"])
#ax.set_title("Beijing 24/11/2023, Particles")

plt.show()
#plt.savefig("beijing_20200501.png", bbox_inches='tight', dpi=300)