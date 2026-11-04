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

di_original=ds["neg_ions"]
di_original=di_original.copy()
di2=di_original.dropna(dim="diameter", how="all")
di=di2.dropna(dim="time", how="all")

dfi=di.to_pandas()

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")
dfi.index = dfi.index.tz_localize("UTC").tz_convert("Europe/Rome")

concentrations = af.calc_conc_interp(df, 2.5e-9, 5e-9).iloc[:, 0]

ion_conc = af.calc_conc_interp(
    dfi,
    1e-9,
    2e-9
).iloc[:, 0]

ion_conc = ion_conc.where(ion_conc > 0)

ion_conc_2 = af.calc_conc_interp(
    dfi,
    2.5e-9,
    5e-9
).iloc[:, 0]

ion_conc_2 = ion_conc_2.where(ion_conc > 0)

cs = 0.9*(2/ion_conc-1.6e-6*ion_conc)

ncs = ion_conc_2 * cs

pvals = np.percentile(ncs, np.arange(1, 101))

pvals = np.array(pvals, dtype=float)

logN = np.log10(pvals)

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    range(1,101),
    pvals
)

ax.set_xlabel(
    'Percentile'
)

ax.set_ylabel(
    'log10N'
)

plt.show()


threshold = np.percentile(ncs, 80)
print(threshold)
