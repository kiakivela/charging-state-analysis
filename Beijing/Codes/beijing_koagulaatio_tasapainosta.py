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

neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

f = neg_i / neg_p

dp_nm = np.array(neg_p.columns, dtype=float)

coagS0_nais = af.calc_coags(neg_p, dp_nm[0:38]).squeeze()
coagS0_nais_med = np.nanmedian(coagS0_nais, axis=0)

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1
dp_nm_2 = dp_nm*1e9
N1 = 300

df = pd.read_excel(r"C:\Users\kivekia\Downloads\Python\BUCT DMPS 2018-2022\BUCT DMPS 2018-2022\DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
df.columns = df.columns.astype(float) * 1e-9

coagN = af.calc_coags(df, dp_nm).squeeze()
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

x = (1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)

#coagN = np.asarray(coagN)
#coagS = np.asarray(coagS)

coagN_med = np.nanmedian(coagN, axis=0)
coagS_med = np.nanmedian(coagS, axis=0)

coagS0 = pd.DataFrame(
    index=f.index,
    columns=f.columns,
    dtype=float
)

for diameter in dp_nm:
    coagS0[diameter] = (
        a0 * (diameter / d0)**l * N1
        - f[diameter] * a * N1
    ) / (
        (x[dp_nm == diameter][0] - 1) * f[diameter]
    )

coagS0_med = np.nanmedian(coagS0, axis=0)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    dp_nm_2,
    coagN_med,
    marker="o",
    markersize=4,
    label="DMPS"
)

ax.plot(
    dp_nm_2,
    coagS0_med,
    marker="o",
    markersize=4,
    label="Equation"
)

#ax.plot(
#    dp_nm_2[0:38],
#    coagS0_nais_med,
#    marker="o",
#    markersize=4,
#    label="NAIS"
#)

ax.set_xlabel("Particle diameter (nm)")
ax.set_ylabel(r"$coagS_0$ (cm$^{-3}$ s$^{-1}$)")
ax.set_title(r"$coagS_0$ vs. particle diameter")
ax.set_xscale("log")

ax.legend()

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
