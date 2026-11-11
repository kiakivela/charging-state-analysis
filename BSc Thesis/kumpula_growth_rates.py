import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata

ds = xr.open_dataset("/home/kivekia/Python/kumpula_1h_checked.nc")

da_original=ds["pos_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

dp_low = 2e-9
dp_high = 3e-9

diams = df.columns.astype(float).values

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
df_2024 = df.loc["2024"]
gr_list1 = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=2,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list1.append(gr["gr"])

import matplotlib.pyplot as plt
gr_array = np.array(gr_list1)
gr_array = gr_array[gr_array > 0]
gr_filtered = gr_array[gr_array >= 0.5]   # typical cutoff 0.3–0.5 nm/h

bins = np.logspace(np.log10(0.1), np.log10(100), 20)

plt.hist(gr_filtered, bins=bins, color="black")
plt.xscale("log")
plt.xlim(0.1, 100)
plt.xlabel("GR (nm h$^{-1}$)")
plt.ylabel("Number of days")
plt.title("Daily growth rates 2024, Positive Particles at 2-3 nm")
#plt.show()

pm=np.mean(gr_filtered)
print(pm)

pmedian=np.median(gr_filtered)
print(pmedian)

# ------------------- Negative Particles 2-3 nm-------------------------

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

dp_low = 2e-9
dp_high = 3e-9

diams = df.columns.astype(float).values

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
df_2024 = df.loc["2024"]
gr_list1 = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=2,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list1.append(gr["gr"])

import matplotlib.pyplot as plt
gr_array = np.array(gr_list1)
gr_array = gr_array[gr_array > 0]
gr_filtered1 = gr_array[gr_array >= 0.5]   # typical cutoff 0.3–0.5 nm/h

bins = np.logspace(np.log10(0.1), np.log10(100), 20)

plt.hist(gr_filtered1, bins=bins, color="black")
plt.xscale("log")
plt.xlim(0.1, 100)
plt.ylim(0, 11)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("GR (nm h$^{-1}$)", fontsize=18)
plt.ylabel("Number of days", fontsize=18)
plt.title("Particles at 2-3 nm", fontsize=18)
plt.show()

# ------------  Negative Particles 3-7 nm ----------

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

dp_low = 3e-9
dp_high = 7e-9

diams = df.columns.astype(float).values

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
df_2024 = df.loc["2024"]
gr_list2 = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=3,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list2.append(gr["gr"])

import matplotlib.pyplot as plt
gr_array = np.array(gr_list2)
gr_array = gr_array[gr_array > 0]
gr_filtered2 = gr_array[gr_array >= 1]   # typical cutoff 0.3–0.5 nm/h

bins = np.logspace(np.log10(0.1), np.log10(100), 20)

plt.hist(gr_filtered2, bins=bins, color="black")
plt.xscale("log")
plt.xlim(0.1, 100)
plt.ylim(0, 11)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("GR (nm h$^{-1}$)", fontsize=18)
plt.ylabel("Number of days", fontsize=18)
plt.title("Particles at 3-7 nm", fontsize=18)
plt.show()

# ------------- Negative Particles 7-20 nm -------------------

da_original=ds["neg_particles"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

dp_low = 7e-9
dp_high = 20e-9

diams = df.columns.astype(float).values

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
df_2024 = df.loc["2024"]
gr_list3 = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=4,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list3.append(gr["gr"])

import matplotlib.pyplot as plt
gr_array = np.array(gr_list3)
gr_array = gr_array[gr_array > 0]
gr_filtered3 = gr_array[gr_array >= 1]   # typical cutoff 0.3–0.5 nm/h

bins = np.logspace(np.log10(0.1), np.log10(100), 20)

plt.hist(gr_filtered3, bins=bins, color="black")
plt.xscale("log")
plt.xlim(0.1, 100)
plt.ylim(0, 11)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("GR (nm h$^{-1}$)", fontsize=18)
plt.ylabel("Number of days", fontsize=18)
plt.title("Particles at 7-20 nm", fontsize=18)
plt.show()

# ---------------- Negative Ions 2-3 nm --------------------

da_original=ds["neg_ions"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

dp_low = 2e-9
dp_high = 3e-9

diams = df.columns.astype(float).values

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
df_2024 = df.loc["2024"]
gr_list4 = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=2,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list4.append(gr["gr"])

import matplotlib.pyplot as plt
gr_array = np.array(gr_list4)
gr_array = gr_array[gr_array > 0]
gr_filtered4 = gr_array[gr_array >= 0.5]   # typical cutoff 0.3–0.5 nm/h

bins = np.logspace(np.log10(0.1), np.log10(100), 20)

plt.hist(gr_filtered4, bins=bins, color="black")
plt.xscale("log")
plt.xlim(0.1, 100)
plt.ylim(0, 11)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("GR (nm h$^{-1}$)", fontsize=18)
plt.ylabel("Number of days", fontsize=18)
plt.title("Negative Ions at 2-3 nm", fontsize=18)
plt.show()

# ---------------- Negative Ions 3-7 nm --------------------

da_original=ds["neg_ions"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

dp_low = 3e-9
dp_high = 7e-9

diams = df.columns.astype(float).values

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
df_2024 = df.loc["2024"]
gr_list5 = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=3,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list5.append(gr["gr"])

import matplotlib.pyplot as plt
gr_array = np.array(gr_list5)
gr_array = gr_array[gr_array > 0]
gr_filtered5 = gr_array[gr_array >= 1]   # typical cutoff 0.3–0.5 nm/h

bins = np.logspace(np.log10(0.1), np.log10(100), 20)

plt.hist(gr_filtered5, bins=bins, color="black")
plt.xscale("log")
plt.xlim(0.1, 100)
plt.ylim(0, 11)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("GR (nm h$^{-1}$)", fontsize=18)
plt.ylabel("Number of days", fontsize=18)
plt.title("Negative Ions at 3-7 nm", fontsize=18)
plt.show()

# ---------------- Negative Ions 7-20 nm --------------------

da_original=ds["neg_ions"]
da_original=da_original.copy()
da2=da_original.dropna(dim="diameter", how="all")
da=da2.dropna(dim="time", how="any")

df=da.to_pandas()

dp_low = 7e-9
dp_high = 20e-9

diams = df.columns.astype(float).values

in_range_idx = np.where((diams > dp_low) & (diams < dp_high))[0]

expanded_idx = np.unique(np.concatenate([in_range_idx - 1, in_range_idx, in_range_idx + 1]))

    # Clip indices to stay within bounds of array
expanded_idx = expanded_idx[(expanded_idx >= 0) & (expanded_idx < len(diams))]

df_in_range = df.iloc[:,expanded_idx]

data_reso=1.0
nan_threshold=20.0
col_threshold=1

df.index = df.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
df_2024 = df.loc["2024"]
gr_list6 = []

for date, df_day in df_2024.groupby(df_2024.index.date):

    # skip short days
    if len(df_day) < 24:
        continue

    gr = af.cross_corr_gr(
        df_day,
        dp_low,
        dp_high,
        smoothing_window=3.0,
        tau_window=23,
        col_threshold=1,
        row_threshold=1,
        number_of_divisions=4,
        data_reso=1.0,
        verbose=False
    )

    if isinstance(gr, dict):
        gr_list6.append(gr["gr"])

import matplotlib.pyplot as plt
gr_array = np.array(gr_list6)
gr_array = gr_array[gr_array > 0]
gr_filtered6 = gr_array[gr_array >= 1]   # typical cutoff 0.3–0.5 nm/h

bins = np.logspace(np.log10(0.1), np.log10(100), 20)

plt.hist(gr_filtered6, bins=bins, color="black")
plt.xscale("log")
plt.xlim(0.1, 100)
plt.ylim(0, 11)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("GR (nm h$^{-1}$)", fontsize=18)
plt.ylabel("Number of days", fontsize=18)
plt.title("Negative Ions at 7-20 nm", fontsize=18)
plt.show()

import numpy as np

print("mean values")
m1=np.mean(gr_filtered1)
print(m1)

m2=np.mean(gr_filtered2)
print(m2)

m3=np.mean(gr_filtered3)
print(m3)

m4=np.mean(gr_filtered4)
print(m4)

m5=np.mean(gr_filtered5)
print(m5)

m6=np.mean(gr_filtered6)
print(m6)

print("median values")

me1=np.median(gr_filtered1)
print(me1)

me2=np.median(gr_filtered2)
print(me2)

me3=np.median(gr_filtered3)
print(me3)

me4=np.median(gr_filtered4)
print(me4)

me5=np.median(gr_filtered5)
print(me5)

me6=np.median(gr_filtered6)
print(me6)

print("90th percentile")

p1=np.percentile(gr_filtered1, 90)
print(p1)

p2=np.percentile(gr_filtered2, 90)
print(p2)

p3=np.percentile(gr_filtered3, 90)
print(p3)

p4=np.percentile(gr_filtered4, 90)
print(p4)

p5=np.percentile(gr_filtered5, 90)
print(p5)

p6=np.percentile(gr_filtered6, 90)
print(p6)