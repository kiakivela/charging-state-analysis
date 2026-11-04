import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
from pathlib import Path

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/beijing/*.nc"))

files = [Path(f) for f in files]

weekend_files = []

for f in files:

    date = pd.to_datetime(
        f.stem.split("_")[1],
       format="%Y%m%d"
        )
    if date.dayofweek >= 5:
       weekend_files.append(f)

datasets = [xr.open_dataset(f) for f in weekend_files]

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

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
dfi.index = dfi.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

concentrations = af.calc_conc_interp(df, 2.5e-9, 5e-9).iloc[:,0]

ion_conc = af.calc_conc_interp(
    dfi,
    1e-9,
    2e-9
).iloc[:,0]

ion_conc = ion_conc.where(ion_conc > 0)

cs = 0.9*(2/ion_conc-1.6e-6*ion_conc)

ncs = concentrations * cs

threshold = np.percentile(ncs, 60)

dfi = dfi.loc[:, dfi.columns >= 2.5e-9]
dfi = dfi.loc[:, dfi.columns <= 11e-9]

df = df.loc[:, df.columns >= 2.5e-9]
df = df.loc[:, df.columns <= 11e-9]

mask = ncs > threshold

df = df.where(mask)
dfi = dfi.where(mask)

common_cols = dfi.columns.intersection(
    df.columns
)

dfi = dfi[common_cols]
df = df[common_cols]

f_meas = dfi / df

#f_meas = f_meas.where(mask)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

f_eq = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in f_meas.columns
    }
)

overcharge = f_meas.divide(
    f_eq,
    axis=1
)
small = overcharge.copy()
#small = small.loc[:, small.columns >=2.5e-9]
small = small.loc[:, small.columns <=3e-9]

large = overcharge.copy()
large = large.loc[:, large.columns >=10e-9]
large = large.loc[:, large.columns <=12e-9]

small_med = small.median(axis=1, numeric_only=True)
large_med = large.median(axis=1, numeric_only=True)

# ratio between size ranges
ratio_series = large_med / small_med

grouped = ratio_series.groupby(
    ratio_series.index.strftime('%H:%M')
)

p25 = grouped.quantile(0.25)
p50 = grouped.quantile(0.50)
p75 = grouped.quantile(0.75)

# convert to datetime
for s in [p25, p50, p75]:
    s.index = pd.to_datetime(s.index, format='%H:%M')

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(p25.index, p25.values, label='25th percentile')
ax.plot(p50.index, p50.values, label='50th percentile')
ax.plot(p75.index, p75.values, label='75th percentile')

ax.set_xlabel("Time of day")
ax.set_ylabel("Ratio")

ax.set_ylim(0, 5)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()