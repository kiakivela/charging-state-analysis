import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
import math

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/milan/*.nc"))
good_files = []
bad_files = []

for f in files:

    ds = xr.open_dataset(f)

    time = pd.to_datetime(ds.time.values)

    # time completeness
    expected = pd.date_range(
        start=time[0].floor("D"),
        periods=144,
        freq="10min"
    )

    time_ok = (
        len(time) == 144
        and pd.DatetimeIndex(time).equals(expected)
    )

    # data completeness
    da = ds["neg_particles"]

    valid_fraction = np.isfinite(da.values).mean()

    data_ok = valid_fraction > 0.95

    if time_ok and data_ok:
        good_files.append(f)
    else:
        bad_files.append(f)

        print(f)
        print("valid fraction:", valid_fraction)

print("Good files:", len(good_files))
print("Bad files:", len(bad_files))
