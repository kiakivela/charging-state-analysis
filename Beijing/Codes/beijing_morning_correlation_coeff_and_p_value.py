import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap

import matplotlib
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = matplotlib.colormaps.get_cmap

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

neg_i[neg_i < 0] = np.nan
neg_p[neg_p < 0] = np.nan

dp_nm = np.array(neg_p.columns, dtype=float)
dp_nm_2 = dp_nm*1e9

df = pd.read_excel(r"C:\Users\kivekia\Downloads\Python\BUCT DMPS 2018-2022\BUCT DMPS 2018-2022\DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

df.index = df.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

neg_i_raw = neg_i.copy()
neg_p_raw = neg_p.copy()

neg_i_hourly = neg_i_raw.resample("1h").median()
neg_p_hourly = neg_p_raw.resample("1h").median()

neg_i_hourly = neg_i_hourly.dropna(how="all")
neg_p_hourly = neg_p_hourly.dropna(how="all")

df_hourly = df.copy()
df_hourly.index = df_hourly.index.floor("1h")
df_hourly = df_hourly.loc[
    ~df_hourly.index.duplicated(keep="first")
]

common_index = (
    neg_i_hourly.index
    .intersection(neg_p_hourly.index)
    .intersection(df_hourly.index)
)

neg_i = neg_i_hourly.loc[common_index].copy()
neg_p = neg_p_hourly.loc[common_index].copy()
df = df_hourly.loc[common_index].copy()

df.columns = df.columns.astype(float) * 1e-9

coagN = af.calc_coags(df, dp_nm)
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)


N = af.calc_conc(neg_i, 0.0, 1.6e-9).squeeze()

#neg_p = neg_p.loc[:, neg_p.columns <= 11e-9]
#neg_i = neg_i.loc[:, neg_i.columns <= 11e-9]

neg_p = neg_p.loc[:, neg_p.columns >= 2.5e-9]
neg_i = neg_i.loc[:, neg_i.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p, 2.5e-9, 5e-9).squeeze()

#print(np.percentile(concentrations, 40))

threshold = 1000

mask = concentrations > threshold

neg_p_masked = neg_p.copy()
neg_i_masked = neg_i.copy()

neg_p_masked[~mask] = np.nan
neg_i_masked[~mask] = np.nan

#neg_p_masked = neg_p_masked.loc[:, neg_p_masked.columns <= 11e-9]
#neg_i_masked = neg_i_masked.loc[:, neg_i_masked.columns <= 11e-9]

neg_p_masked = neg_p_masked.loc[:, neg_p_masked.columns >= 2.5e-9]
neg_i_masked = neg_i_masked.loc[:, neg_i_masked.columns >= 2.5e-9]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

f_meas = neg_i / neg_p

f_eq_1 = pd.Series(
    {
        diameter: af.eq_charge_frac(diameter, -1)
        for diameter in neg_p.columns
    }
)

overcharge_1 = f_meas/f_eq_1

S_ranges = {
    "3-4":  (3e-9, 4e-9),
    "4-6":  (4e-9, 6e-9),
    "8-10": (8e-9, 10e-9),
    "10-15": (10e-9, 15e-9),
    "15-20": (15e-9, 20e-9),
    "20-25": (20e-9, 25e-9)
}

N_ranges = {
    "3-4":  (3e-9, 4e-9),
    "4-6":  (4e-9, 6e-9),
    "8-10": (8e-9, 10e-9),
    "10-15": (10e-9, 15e-9),
    "15-20": (15e-9, 20e-9),
    "20-25": (20e-9, 25e-9)
}

S_data = {}

for name, (d_min, d_max) in S_ranges.items():

    S_subset = overcharge_1.loc[
        :,
        (overcharge_1.columns >= d_min) &
        (overcharge_1.columns <= d_max)
    ]

    S_data[name] = S_subset.median(axis=1)

N_data = {}

for name, (d_min, d_max) in N_ranges.items():

    N_data[name] = af.calc_conc(
        neg_p,
        d_min,
        d_max
    ).squeeze()

start = "2020-10-01 00:00:00"
end = "2022-06-29 23:59:59"

for name in S_data:
    S_data[name] = S_data[name].loc[start:end]

for name in N_data:
    N_data[name] = N_data[name].loc[start:end]

from scipy.stats import pearsonr

results = []

for N_name, N_series in N_data.items():

    for S_name, S_series in S_data.items():

        # Combine S and N using their common timestamps
        data = pd.concat(
            [
                S_series.rename("S"),
                N_series.rename("N")
            ],
            axis=1
        )

        # Keep only 04:00–08:00
        #time_mask = (
        #    (data.index.hour >= 4) &
        #    (data.index.hour < 8)
        #)
        #data = data.loc[time_mask]

        # Remove NaN and infinite values
        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        # Only positive values can be log10 transformed
        data = data[
           (data["S"] > 0) &
            (data["N"] > 0)
        ]

        # Pearson correlation on log-transformed data
        r, p = pearsonr(
            np.log10(data["S"]),
            np.log10(data["N"])
        )

        results.append({
            "S": S_name,
            "N": N_name,
            "Correlation": r,
            "P-value": p,
            "N points": len(data)
        })

results_df = pd.DataFrame(results)

correlation_matrix = results_df.pivot(
    index="S",
    columns="N",
    values="Correlation"
)

pvalue_matrix = results_df.pivot(
    index="S",
    columns="N",
    values="P-value"
)

with pd.ExcelWriter(
    r"C:\Users\kivekia\Downloads\correlation_results.xlsx"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="All results",
        index=False
    )

    correlation_matrix.to_excel(
        writer,
        sheet_name="Correlation"
    )

    pvalue_matrix.to_excel(
        writer,
        sheet_name="P-values"
    )

