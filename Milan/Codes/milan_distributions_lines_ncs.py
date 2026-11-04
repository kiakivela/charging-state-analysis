import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
import math
import matplotlib.dates as mdates

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/milan/NAIS_20230406.nc")

#files = sorted(glob.glob("/home/kivekia/Summer Work 2026/Milan/Apr 2026/*.nc"))
#datasets = [xr.open_dataset(f) for f in files]
#ds = xr.concat(datasets, dim='time')
#ds = ds.sortby('time')

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="all")

di_original=ds["neg_ions"]
di_original=di_original.copy()
di2=di_original.dropna(dim="diameter", how="all")
di=di2.dropna(dim="time", how="all")

dfi=di.to_pandas()
df=da.to_pandas()

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")
dfi.index = dfi.index.tz_localize("UTC").tz_convert("Europe/Rome")

concentrations = af.calc_conc(df, 2.5e-9, 5e-9).iloc[:,0]

ion_conc = af.calc_conc_interp(
    dfi,
    1e-9,
    2e-9
).iloc[:,0]

ion_conc = ion_conc.where(ion_conc > 0)

cs = 0.9*(2/ion_conc-1.6e-6*ion_conc)

ncs = concentrations * cs

threshold = np.percentile(ncs, 90)

#dfi = dfi.loc[:, dfi.columns >= 2.5e-9]
#dfi = dfi.loc[:, dfi.columns <= 11e-9]

#f = df.loc[:, df.columns >= 2.5e-9]
#df = df.loc[:, df.columns <= 11e-9]

df_cs = df.mul(cs, axis=0)

mask = (df_cs > threshold).iloc[:,0]

event_times_horizontal = df.index[mask]

#df = df.loc[:, df.columns <= 11e-9]
#df.loc[:, df.columns <= 2e-9] = np.nan

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    df,
    ax,
    norm="log",
    clim=[1e3, 1e6]
)


for t in event_times_horizontal:
    ax.hlines(
        y=np.log10(4e-9),
        xmin=t - pd.Timedelta("10min"),
        xmax=t + pd.Timedelta("10min"),
        color='black',
        linewidth=2,
        zorder=10
    )

#for t in event_times: 
# ax.axvline(t, color='black', linewidth=1)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

cbar.set_ticks([1e3, 1e4, 1e5])
cbar.set_ticklabels(["10³", "10⁴", "10⁵"])

plt.savefig("milan_extreme_lines_20230406_set_threshold_ncs.png", bbox_inches="tight", dpi=300)