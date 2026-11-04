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

threshold = 22

for f in files:

    ds = xr.open_dataset(f)

    da = ds["neg_particles"]
    di = ds["neg_ions"]

    da = da.dropna(dim="diameter", how="all")
    da = da.dropna(dim="time", how="all")
    di = di.dropna(dim="diameter", how="all")
    di = di.dropna(dim="time", how="all")

    df = da.to_pandas()
    dfi = di.to_pandas()

    conc = af.calc_conc(
        df,
        2.5e-9,
        5e-9
    ).squeeze()

    ion_conc = af.calc_conc_interp(
    dfi,
    1e-9,
    2e-9
    ).squeeze()

    ion_conc = ion_conc.where(ion_conc > 0)

    cs = 0.9*(2/ion_conc-1.6e-6*ion_conc)

    ncs = conc * cs

    if ncs.max() >= threshold:
        event_files.append(f)
        event_strengths[f] = np.percentile(ncs, 90)
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