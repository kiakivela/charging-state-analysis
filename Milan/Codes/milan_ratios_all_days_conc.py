import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
from pathlib import Path

ds = xr.open_dataset(r"C:\Users\kivekia\Downloads\Python\milan_all.nc")

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

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Europe/Rome")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Europe/Rome")

concentrations = af.calc_conc(neg_p, 2.5e-9, 5e-9).squeeze()

threshold = 1250

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

d1=10e-9
d2=12e-9

d3=20e-9
d4=24e-9

conc_neg_p_small = af.calc_conc(neg_p_masked, d1, d2, 1)
conc_neg_i_small=af.calc_conc(neg_i_masked, d1, d2, 1)

conc_neg_p_large = af.calc_conc(neg_p_masked, d3, d4, 1)
conc_neg_i_large=af.calc_conc(neg_i_masked, d3, d4, 1)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

f_meas_small = conc_neg_i_small.divide(conc_neg_p_small, axis=1)
f_meas_large = conc_neg_i_large.divide(conc_neg_p_large, axis=1)

f_eq1=af.eq_charge_frac((d1+d2)/2, -1)
f_eq2=af.eq_charge_frac((d3+d4)/2, -1)

overcharge1 = f_meas_small.divide(
    f_eq1,
    axis=1
)

overcharge2 = f_meas_large.divide(
    f_eq2,
    axis=1
)

# ratio between size ranges
ratio_series = overcharge1 / overcharge2

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

#ax.set_ylim(0, 5)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()