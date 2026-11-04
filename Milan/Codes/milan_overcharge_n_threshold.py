import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/milan/*.nc"))
datasets = [xr.open_dataset(f) for f in files]

ds = xr.concat(datasets, dim='time')

ds = ds.sortby('time')

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()

negp_og2=negp_og.dropna(dim="diameter", how="all")
negp=negp_og2.dropna(dim="time", how="all")

negi_og2=negi_og.dropna(dim="diameter", how="all")
negi=negi_og2.dropna(dim="time", how="all")

neg_p=negp.to_pandas()
neg_i=negi.to_pandas()

common_index = neg_p.index.intersection(neg_i.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i.loc[common_index]

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Europe/Rome")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Europe/Rome")

concentrations = af.calc_conc_interp(neg_p, 2.5e-9, 5e-9).squeeze()

threshold = 2200

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

#df_events.loc[:, df_events.columns < 2.5e-9] = np.nan
#df_events.loc[:, df_events.columns > 5e-9] = np.nan

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# measured charging fraction
f_meas = neg_i_masked / neg_p_masked

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

plt.savefig("milan_overcharge_n_threshold", bbox_inches='tight', dpi=300)