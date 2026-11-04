import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
from pathlib import Path

ds = xr.open_dataset(r"C:\Users\kivekia\Downloads\Python\beijing_all.nc")

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

#neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
#neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

#neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
#neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

concentrations = af.calc_conc(neg_p, 3e-9, 10e-9).squeeze()
#concentrations = pd.to_numeric(concentrations, errors="coerce")

start = "2020-10-01 00:00:00"
end = "2022-06-29 23:59:59"

concentrations = concentrations.loc[start:end]

grouped = concentrations.groupby(
    concentrations.index.strftime('%H:%M')
)

median = grouped.median()

median.index = pd.to_datetime(
    median.index,
    format='%H:%M'
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    median.index,
    median.values,
    color='black',
    label='Median'
)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ax.set_xlabel("Time of day")
ax.set_ylabel("Concentration")

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

p5 = grouped.quantile(0.05)
p10 = grouped.quantile(0.10)
p15 = grouped.quantile(0.15)
p20 = grouped.quantile(0.20)
p25 = grouped.quantile(0.25)
p30 = grouped.quantile(0.30)
p35 = grouped.quantile(0.35)
p40 = grouped.quantile(0.40)
p45 = grouped.quantile(0.45)
p50 = grouped.quantile(0.50)
p55 = grouped.quantile(0.55)
p60 = grouped.quantile(0.60)
p65 = grouped.quantile(0.65)
p70 = grouped.quantile(0.70)
p75 = grouped.quantile(0.75)
p80 = grouped.quantile(0.80)
p85 = grouped.quantile(0.85)
p90 = grouped.quantile(0.90)
p95 = grouped.quantile(0.95)

# convert to datetime
for s in [p5, p10, p15, p20, p25, p30, p35, p40, p45, p50, p55, p60, p65, p70, p75, p80, p85, p90, p95]:
    s.index = pd.to_datetime(s.index, format='%H:%M')

fig, ax = plt.subplots(figsize=(10,5))

#ax.plot(p5.index, p5.values, label='5th percentile')
#ax.plot(p10.index, p10.values, label='10th percentile')
#ax.plot(p15.index, p15.values, label='15th percentile')
#ax.plot(p20.index, p20.values, label='20th percentile')
#ax.plot(p25.index, p25.values, label='25th percentile')
#ax.plot(p30.index, p30.values, label='30th percentile')
#ax.plot(p35.index, p35.values, label='35th percentile')
#ax.plot(p40.index, p40.values, label='40th percentile')
#ax.plot(p45.index, p45.values, label='45th percentile')
#ax.plot(p50.index, p50.values, label='50th percentile')
#ax.plot(p55.index, p55.values, label='55th percentile')
#ax.plot(p60.index, p60.values, label='60th percentile')
#ax.plot(p65.index, p65.values, label='65th percentile')
#ax.plot(p70.index, p70.values, label='70th percentile')
#ax.plot(p75.index, p75.values, label='75th percentile')
#ax.plot(p80.index, p80.values, label='80th percentile')
#ax.plot(p85.index, p85.values, label='85th percentile')
#
# ax.plot(p95.index, p95.values, label='95th percentile')

ax.set_xlabel("Time of day")
ax.set_ylabel("")

#ax.set_ylim(0, 15)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
#plt.show()

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(p50.index, p50.values, color='black', label='median')

ax.fill_between(
    p25.index,
    p25.values,
    p75.values,
    alpha=0.3,
    label='25–75%'
)

ax.fill_between(
    p10.index,
    p90.values,
    p10.values,
    alpha=0.15,
    label='10–90%'
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M')
)

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
#plt.show()
