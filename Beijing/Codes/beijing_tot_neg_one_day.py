import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap

ds = xr.open_dataset("/home/kivekia/Summer Work 2026/beijing/NAIS_20181230.nc")

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

#neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
#neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]
#pos_p = pos_p.loc[:, pos_p.columns <= 11e-9]
#pos_i = pos_i.loc[:, pos_i.columns <= 11e-9]
# 
#neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
#neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]
#pos_p = pos_p.loc[:, pos_p.columns >= 2.5e-9]
#pos_i = pos_i.loc[:, pos_i.columns >= 2.5e-9]

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
pos_p.index = pos_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
pos_i.index = pos_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


ratio = neg_p / neg_i

#df_30 = ratio.resample('30min').median()

#df_30["clocktime"] = df_30.index.strftime('%H:%M')

#median_diurnal = (
#    df_30
#    .groupby("clocktime")
#    .median(numeric_only=True)
#)

#times = sorted(median_diurnal.index)

#median_diurnal = median_diurnal.loc[times]


#median_diurnal.index = pd.to_datetime(
#    median_diurnal.index,
#    format='%H:%M')

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    ratio,
    ax,
    norm='linear',
)

#ax.xaxis.set_major_formatter(
#    mdates.DateFormatter('%H:%M'))

cbar.set_label('Neutral Particles / Negative particles')

plt.savefig("ratio_beijing_neut_neg_20181230.png", bbox_inches='tight', dpi=300)