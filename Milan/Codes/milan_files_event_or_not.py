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
event_files = []
non_event_files = []
event_strengths = {}

threshold = 9880

for f in files:

    ds = xr.open_dataset(f)

    da = ds["neg_particles"]

    da = da.dropna(dim="diameter", how="all")
    da = da.dropna(dim="time", how="all")

    df = da.to_pandas()

    conc = af.calc_conc(
        df,
        2.5e-9,
        5e-9
    ).squeeze()


    if conc.max() >= threshold:
        event_files.append(f)
        event_strengths[f] = np.percentile(conc, 97)
    else:
        non_event_files.append(f)

top_days = sorted(
    event_strengths.items(),
    key=lambda x: x[1],
    reverse=True
)

for day, value in top_days[:10]:
    print(day, value)

print("Event days:", len(event_files))
print("Non-event days:", len(non_event_files))