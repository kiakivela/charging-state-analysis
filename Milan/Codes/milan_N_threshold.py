import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
import math

#ds = xr.open_dataset("/home/kivekia/Summer Work 2026/milan/NAIS_20260213.nc")

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/milan/*.nc"))
datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim='time')
ds = ds.sortby('time')

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="all")

df=da.to_pandas()

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")

concentrations = af.calc_conc_interp(df, 2.5e-9, 5e-9).iloc[:, 0]

concentrations = concentrations.where(concentrations > 0)
concentrations = concentrations.dropna()

pvals = np.percentile(concentrations, np.arange(1, 101))

pvals = np.array(pvals, dtype=float)

logN = np.log10(pvals)

dlogN = np.diff(logN)

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    range(1,101),
    logN
)

ax.set_xlabel(
    'Percentile'
)

ax.set_ylabel(
    'log10N'
)

#plt.show()


threshold = np.percentile(concentrations, 80)
print(threshold)

