import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.lab as al

ds = xr.open_dataset("/home/kivekia/Python/beijing_all.nc")

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
l = 1.2
dp_nm = np.arange(1, 101)*1e-9
dp_nm_2 = dp_nm*1e9
N1 = 300

df = pd.read_excel("/home/kivekia/Python/BUCT DMPS 2018-2022/BUCT DMPS 2018-2022/DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
df.columns = df.columns.astype(float) * 1e-9

coagN = af.calc_coags(df, dp_nm).squeeze()
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

coagN_med = np.nanmedian(coagN, axis=0)
coagS_med = np.nanmedian(coagS, axis=0)

f_eq_1 = pd.Series(
    {
        diameter: al.eq_charge_frac(diameter, -1)
        for diameter in dp_nm
    }
)

f_eq_2 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)/a
        for diameter in dp_nm
    }
)

f_eq_3 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_3[i] = (
        a0*(diameter/d0)**l * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
    )

df_2 = df.copy()
df_2 = df_2 / 2
coagN_2 = af.calc_coags(df_2, dp_nm).squeeze()
coagS_2 = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN_2)

coagN_med_2 = np.nanmedian(coagN_2, axis=0)
coagS_med_2 = np.nanmedian(coagS_2, axis=0)

f_eq_4 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_4[i] = (
        a0*(diameter/d0)**l * N1
    ) / (
        a*N1
        + coagS_med_2[i]
        - coagN_med_2[i]
    )

df_3 = df_2.copy()
df_3 = df_3 / 2
coagN_3 = af.calc_coags(df_3, dp_nm).squeeze()
coagS_3 = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN_3)

coagN_med_3 = np.nanmedian(coagN_3, axis=0)
coagS_med_3 = np.nanmedian(coagS_3, axis=0)

f_eq_5 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_5[i] = (
        a0*(diameter/d0)**l * N1
    ) / (
        a*N1
        + coagS_med_3[i]
        - coagN_med_3[i]
    )

calc1 =  f_eq_2.values - f_eq_3
calc2 = f_eq_1.values - calc1

calc3 = f_eq_2.values - f_eq_4
calc4 = f_eq_1.values - calc3

calc5 = f_eq_2.values - f_eq_5
calc6 = f_eq_1.values - calc5

# Diameter in nm for plotting
diam_nm = np.arange(1, 101)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis
line1 = ax1.plot(
    diam_nm,
    calc2,
    label="Equilibrium fraction",
)

line2 = ax1.plot(
    diam_nm,
    calc4,
   label="Equilibrium fraction, koag / 2",
)

line3 = ax1.plot(
    diam_nm,
    calc6,
    label="Equilibrium fraction, koag / 4",
)

ax1.set_xlabel("Diameter (nm)")
ax1.set_ylabel("Equilibrium Charge Fraction")
ax1.set_xscale("log")
ax1.set_yscale("log")
#ax1.set_xlim(2e-9, 20e-9)
#ax1.set_ylim(-0.05, 0.9)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()

ax1.legend(
    handles1,
    labels1,
    loc="best",
)

plt.tight_layout()
plt.show()