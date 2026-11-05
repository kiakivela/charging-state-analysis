import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob

files = sorted(glob.glob("/home/kivekia/Summer Work 2026/Hyytiälä/*.nc"))
datasets = [xr.open_dataset(f) for f in files]

ds = xr.concat(datasets, dim='time')

ds = ds.sortby('time')

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]
posp_og=ds["pos_particles"]
posi_og=ds["pos_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()
posp_og=posp_og.copy()
posi_og=posi_og.copy()

negp_og2=negp_og.dropna(dim="diameter", how="all")
negp=negp_og2.dropna(dim="time", how="all")

negi_og2=negi_og.dropna(dim="diameter", how="all")
negi=negi_og2.dropna(dim="time", how="all")

posp_og2=posp_og.dropna(dim="diameter", how="all")
posp=posp_og2.dropna(dim="time", how="all")

posi_og2=posi_og.dropna(dim="diameter", how="all")
posi=posi_og2.dropna(dim="time", how="all")

neg_p=negp.to_pandas()
neg_i=negi.to_pandas()
pos_p=posp.to_pandas()
pos_i=posi.to_pandas()

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)
pos_p.columns = pos_p.columns.astype(float)
pos_i.columns = pos_i.columns.astype(float)

neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]
pos_p = pos_p.loc[:, pos_p.columns <= 11e-9]
pos_i = pos_i.loc[:, pos_i.columns <= 11e-9]
# 
neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]
pos_p = pos_p.loc[:, pos_p.columns >= 2.5e-9]
pos_i = pos_i.loc[:, pos_i.columns >= 2.5e-9]

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
pos_p.index = pos_p.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
pos_i.index = pos_i.index.tz_localize("UTC").tz_convert("Europe/Helsinki")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

neutral = neg_p - pos_i - neg_i

ratio = neg_i/neutral

df_30 = ratio.resample('30min').median()

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
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))

cbar.set_label('Negative Particles / Neutral particles')

plt.savefig("ratio_hyytiala_neg_neut_jul_2017.png", bbox_inches='tight', dpi=300)





f_meas = neg_i/(neutral)

background = f_meas.between_time('00:00', '04:00')

f_eq = background.median()

overcharge = f_meas.divide(f_eq, axis=1)

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
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))

cbar.set_label('Overcharge ratio (compared to background)')

plt.savefig("ratio_hyytiala_overcharge_jul_2017", bbox_inches='tight', dpi=300)