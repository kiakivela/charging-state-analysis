import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/beijing/*.nc"))
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

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
dfi.index = dfi.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

concentrations = af.calc_conc_interp(df, 2.5e-9, 5e-9).iloc[:,0]

ion_conc = af.calc_conc_interp(
    dfi,
    1e-9,
    2e-9
).iloc[:,0]

ion_conc = ion_conc.where(ion_conc > 0)

ion_conc_2 = af.calc_conc_interp(
    dfi,
    2.5e-9,
    5e-9
).iloc[:,0]

ion_conc_2 = ion_conc_2.where(ion_conc_2 > 0)

cs = 0.9*(2/ion_conc-1.6e-6*ion_conc)

ncs = ion_conc_2 * cs

threshold = np.percentile(ncs, 80)

dfi = dfi.loc[:, dfi.columns >= 2.5e-9]
dfi = dfi.loc[:, dfi.columns <= 11e-9]

df = df.loc[:, df.columns >= 2.5e-9]
df = df.loc[:, df.columns <= 11e-9]

mask = ncs > threshold

df = df.where(mask, np.nan)
dfi = dfi.where(mask, np.nan)

# measured charging fraction
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

df_30 = overcharge.resample('30min').median()

df_30 = df_30.dropna(how="all")

df_30["clocktime"] = df_30.index.strftime('%H:%M')

median_diurnal = (
    df_30
    .groupby("clocktime")
    .median(numeric_only=True)
)

times = sorted(median_diurnal.index)

median_diurnal = median_diurnal.loc[times]


median_diurnal.index = pd.to_datetime(
    median_diurnal.index,
    format='%H:%M')

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    median_diurnal,
    ax,
    norm='linear',
    clim=[0.28, 0.60]
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))


cbar.set_ticks([0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
cbar.set_ticklabels(["0.30", "0.35", "0.40", "0.45", "0.50", "0.55"])
cbar.set_label('Overcharge ratio')

plt.savefig("beijing_overcharge_ion_ncs_threshold_80", bbox_inches='tight', dpi=300)