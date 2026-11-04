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
l = 1
l_2 = 1.2
l_3 = 1.5
dp_nm = np.arange(1, 101)*1e-9
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

#coagN = np.asarray(coagN)
#coagS = np.asarray(coagS)

coagN_med = np.nanmedian(coagN, axis=0)
coagS_med = np.nanmedian(coagS, axis=0)

f_eq_1 = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in dp_nm
    }
)

f_eq_2 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)/a
        for diameter in dp_nm
    }
)

f_eq_3 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)/(a-2*(a0*(diameter / d0)**l))
        for diameter in dp_nm
    }
)

f_eq_8 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l_2)/(a-2*(a0*(diameter / d0)**l_2))
        for diameter in dp_nm
    }
)


f_eq_9 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l_3)/(a-2*(a0*(diameter / d0)**l_3))
        for diameter in dp_nm
    }
)

f_eq_4 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_4[i] = (
        a0*(diameter/d0)**l * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
    )

f_eq_10 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_10[i] = (
        a0*(diameter/d0)**l_2 * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
    )

f_eq_11 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_11[i] = (
        a0*(diameter/d0)**l_3 * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
    )

f_eq_5 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_5[i] = (
        a0*(diameter/d0)**l * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*a0*(diameter/d0)**l*N1
    )

f_eq_12 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_12[i] = (
        a0*(diameter/d0)**l_2 * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*a0*(diameter/d0)**l_2*N1
    )

f_eq_13 = np.empty(len(dp_nm))

for i, diameter in enumerate(dp_nm):
    f_eq_13[i] = (
        a0*(diameter/d0)**l_3 * N1
    ) / (
        a*N1
        + coagS_med[i]
        - coagN_med[i]
        - 2*a0*(diameter/d0)**l_3*N1
    )

f_eq_6 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l_2)/a
        for diameter in dp_nm
    }
)

f_eq_7 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l_3)/a
        for diameter in dp_nm
    }
)

# Diameter in nm for plotting
diam_nm = np.arange(1, 101)

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
   label="Equilibrium fraction, Vellu, l=1",
)

line3 = ax1.plot(
    diam_nm,
    f_eq_3.values,
    label="Equilibrium fraction, Vellu+(-2a0N), l=1",
)

#line8 = ax1.plot(
#    diam_nm,
#    f_eq_8.values,
#   label="Equilibrium fraction, Vellu+(-2a0N), l=1.2",
#)

#line9 = ax1.plot(
#    diam_nm,
#   f_eq_9.values,
#    label="Equilibrium fraction, Vellu+(-2a0N), l=1.5",
#)

line4 = ax1.plot(
    diam_nm,
    f_eq_4,
    label="Equilibrium fraction, Koagulaatio, l=1",
)

#line10 = ax1.plot(
#    diam_nm,
#    f_eq_10,
#    label="Equilibrium fraction, Koagulaatio, l=1.2",
#)

#line11 = ax1.plot(
#    diam_nm,
#    f_eq_11,
#    label="Equilibrium fraction, Koagulaatio, l=1.5",
#)

line5 = ax1.plot(
    diam_nm,
    f_eq_5,
    label="Equilibrium fraction, Koagulaatio+(-2a0N), l=1",
)

#line12 = ax1.plot(
#    diam_nm,
#    f_eq_12,
#    label="Equilibrium fraction, Koagulaatio+(-2a0N), l=1.2",
#)

#line13 = ax1.plot(
#    diam_nm,
#    f_eq_12,
#    label="Equilibrium fraction, Koagulaatio+(-2a0N), l=1.5",
#)


#line6 = ax1.plot(
#    diam_nm,
#    f_eq_6.values,
#    label="Equilibrium fraction, Vellu. l=1.2",
#)

#line7 = ax1.plot(
#    diam_nm,
#    f_eq_7.values,
#    label="Equilibrium fraction, Vellu. l=1.5",
#)

ax1.set_xlabel("Diameter (m)")
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