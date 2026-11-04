import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt

ds = xr.open_dataset(r"C:\Users\kivekia\Downloads\Python\beijing_all.nc")

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

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.5

f_eq_1 = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in neg_p.columns
    }
)

f_eq_2 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)/a
        for diameter in neg_p.columns
    }
)

# Diameter in nm for plotting
diam_nm = np.array(neg_p.columns, dtype=float)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
line1 = ax1.plot(
    diam_nm,
    f_eq_1.values,
    label="Equilibrium fraction, Wiedensohler",
)

line2 = ax1.plot(
    diam_nm,
    f_eq_2.values,
    label="Equilibrium fraction, Vellu",
)


ax1.set_xlabel("Diameter (m)")
ax1.set_ylabel("Charge fraction")
ax1.set_xscale("log")
ax1.set_xlim(2e-9, 30e-9)
ax1.set_ylim(0, 0.70)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()