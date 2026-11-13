
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import aerosol.functions as af

# --------------------------------------------------
# 1. Read NAIS data
# --------------------------------------------------

ds = xr.open_dataset("/home/kivekia/Python/beijing_all.nc")
ds = ds.sortby("time")

nais = ds["neg_particles"].to_pandas()

nais.columns = nais.columns.astype(float)

# NAIS timestamps: UTC -> Beijing local time
nais.index = (
    nais.index
    .tz_localize("UTC")
    .tz_convert("Asia/Shanghai")
)

nais = nais.sort_index()

# --------------------------------------------------
# 2. Read DMPS data
# --------------------------------------------------

dmps = pd.read_excel(
    "/home/kivekia/Python/DMPS_BUCT_to_D_concentration.xlsx",
    index_col=0,
    parse_dates=True
)

dmps.index = pd.to_datetime(dmps.index)

# DMPS timestamps are already Beijing local time
if dmps.index.tz is None:
    dmps.index = dmps.index.tz_localize("Asia/Shanghai")

dmps.columns = dmps.columns.astype(float)

dmps = dmps.sort_index()
dmps = dmps.loc[~dmps.index.duplicated(keep="first")]

# Convert DMPS diameter columns from nm to metres
dmps.columns = dmps.columns * 1e-9

# --------------------------------------------------
# 3. Calculate 10-15 nm concentrations separately
# --------------------------------------------------

dp_min = 10e-9
dp_max = 15e-9

# NAIS concentration
nais_conc = af.calc_conc_interp(
    nais,
    dp_min,
    dp_max
).squeeze()

# DMPS concentration
dmps_conc = af.calc_conc_interp(
    dmps,
    dp_min,
    dp_max
).squeeze()

# Preserve the original timestamps
nais_conc = pd.Series(
    np.asarray(nais_conc).squeeze(),
    index=nais.index
)

dmps_conc = pd.Series(
    np.asarray(dmps_conc).squeeze(),
    index=dmps.index
)

nais_hourly = nais_conc.resample("1h").median()
dmps_hourly = dmps_conc.resample("1h").median()

# Keep only hours between 01:00 and 03:00
nais_hourly = nais_hourly[
    (nais_hourly.index.hour >= 1) &
    (nais_hourly.index.hour < 3)
]

dmps_hourly = dmps_hourly[
    (dmps_hourly.index.hour >= 1) &
    (dmps_hourly.index.hour < 3)
]

hourly_comparison = pd.concat(
    [
        nais_hourly.rename("NAIS"),
        dmps_hourly.rename("DMPS")
    ],
    axis=1
).dropna()

hourly_comparison = hourly_comparison[
    (hourly_comparison > 0).all(axis=1)
]

# --------------------------------------------------
# 4. Apply the NAIS concentration threshold
# --------------------------------------------------

concentrations = af.calc_conc_interp(
    nais,
    4e-9,
    6e-9
).squeeze()

threshold = 1315

# Calculate hourly median of the threshold concentration
concentrations_hourly = concentrations.resample("1h").median()

# Apply threshold to hourly data
comparison = hourly_comparison.copy()

comparison["threshold_concentration"] = (
    concentrations_hourly.reindex(comparison.index)
)

# Retain only hours where the threshold is exceeded
comparison = comparison[
    comparison["threshold_concentration"] > threshold
].copy()

# Remove threshold helper column
comparison = comparison.drop(columns="threshold_concentration")

# Remove invalid values
comparison = comparison.replace(
    [np.inf, -np.inf], np.nan
).dropna()

comparison = comparison[
    (comparison["NAIS"] > 0) &
    (comparison["DMPS"] > 0)
]

print("Measurements before threshold:", len(hourly_comparison))
print("Measurements after threshold:", len(comparison))

# --------------------------------------------------
# 6. Plot DMPS vs NAIS
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    hourly_comparison["NAIS"],
    hourly_comparison["DMPS"],
    s=8,
    alpha=0.35
)

limits = [
    min(hourly_comparison.min()),
    max(hourly_comparison.max())
]

ax.plot(
    limits,
    limits,
    "k--",
    label="1:1 line"
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("NAIS concentration (cm$^{-3}$)")
ax.set_ylabel("DMPS concentration (cm$^{-3}$)")

ax.set_title(
    f"10–15 nm particle concentration (NAIS threshold > {threshold})"
)

ax.legend()
ax.grid(True, which="both", alpha=0.25)

plt.tight_layout()
plt.savefig("DMPS_vs_NAIS_10_15nm_threshold.png", dpi=300)
plt.show()

# --------------------------------------------------
# 7. Calculate correlation
# --------------------------------------------------

correlation = comparison["NAIS"].corr(
    comparison["DMPS"]
)

print(f"Pearson correlation: {correlation:.3f}")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    comparison.index,
    comparison["NAIS"],
    label="NAIS",
    alpha=0.8
)

ax.plot(
    comparison.index,
    comparison["DMPS"],
    label="DMPS",
    alpha=0.8
)

ax.set_ylabel("Concentration (cm$^{-3}$)")
ax.set_xlabel("Date")
ax.set_title("10–15 nm particle concentration")
ax.set_yscale("log")

ax.legend()
ax.grid(alpha=0.3)

plt.savefig("DMPS_vs_NAIS_10_15nm_timeseries.png", dpi=300)