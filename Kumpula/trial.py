import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap

# 1

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/Kumpula/NAIS_20170227.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 10e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ap.plot_aerosol_dist(df, ax)

plt.savefig("plot31.png", dpi=300)

#2

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/Kumpula/NAIS_20170228.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 10e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ap.plot_aerosol_dist(df, ax)

plt.savefig("plot32.png", dpi=300)

#3

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/Kumpula/NAIS_20170301.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 10e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ap.plot_aerosol_dist(df, ax)

plt.savefig("plot33.png", dpi=300)

#4

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/Kumpula/NAIS_20170302.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 10e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ap.plot_aerosol_dist(df, ax)

plt.savefig("plot34.png", dpi=300)

#5

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/Kumpula/NAIS_20170303.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 10e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ap.plot_aerosol_dist(df, ax)

plt.savefig("plot35.png", dpi=300)

#6

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/Kumpula/NAIS_20170304.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 10e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ap.plot_aerosol_dist(df, ax)

plt.savefig("plot36.png", dpi=300)

#7 

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/Kumpula/NAIS_20170305.nc")

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

#df.columns = df.columns.astype(float)

#df = df.loc[:, df.columns <= 10e-9]

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()
ap.plot_aerosol_dist(df, ax)

plt.savefig("plot37.png", dpi=300)