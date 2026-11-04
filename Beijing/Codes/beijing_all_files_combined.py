import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
from pathlib import Path

files = sorted(glob.glob(r"C:\Users\kivekia\Downloads\Python\Summer Work 2026\milan\*.nc"))

files = [Path(f) for f in files]

weekend_files = []

for f in files:

    date = pd.to_datetime(
        f.stem.split("_")[1],
       format="%Y%m%d"
        )
    if date.dayofweek < 5:
       weekend_files.append(f)

ds = xr.open_mfdataset(
    weekend_files,
    combine="nested",
    concat_dim="time",
    engine="h5netcdf",
    parallel=True
)

ds.to_netcdf("milan_weekdays.nc")