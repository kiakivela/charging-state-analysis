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

common_index = df.index.intersection(dfi.index)

df = df.loc[common_index]
dfi = dfi.loc[common_index]

df.columns = df.columns.astype(float)
dfi.columns = dfi.columns.astype(float)

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")
dfi.index = dfi.index.tz_localize("UTC").tz_convert("Europe/Rome")

#particle_25_5 = df.loc[:, (df.columns >= 2.5e-9) & (df.columns <= 5e-9)]
#ion_25_5 = dfi.loc[:, (dfi.columns >= 2.5e-9) & (dfi.columns <= 5e-9)]

particle_25_5 = af.calc_conc_interp(df, 2.5e-9, 5e-9).iloc[:, 0]
ion_25_5 = af.calc_conc_interp(dfi, 2.5e-9, 5e-9).iloc[:, 0]

# Median concentration for each timestamp
#particle_median = particle_25_5.median(axis=1)
#ion_median = ion_25_5.median(axis=1)

plt.figure(figsize=(8, 6))

plt.scatter(
    ion_25_5,
    particle_25_5,
    label="Particles"
)

plt.yscale("log")

plt.ylabel("Median concentration (2.5–5 nm)")
plt.xlabel("Time")
plt.title("2.5–5 nm Ion and Particle Concentrations")

# Log scales are usually useful for NAIS data
plt.xscale("log")
plt.yscale("log")

plt.legend()
plt.tight_layout()
plt.show()