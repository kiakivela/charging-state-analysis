import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import aerosol.lab as al

import matplotlib
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = matplotlib.colormaps.get_cmap

import xarray as xr
import cfgrib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dk = xr.open_dataset("/home/kivekia/Python/wind_data.grib", engine="cfgrib")

u = dk["u10"].sel(
    latitude=40.0,
    longitude=116.3,
    method="nearest"
)

v = dk["v10"].sel(
    latitude=40.0,
    longitude=116.3,
    method="nearest"
)

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

concentration = af.calc_conc_interp(neg_p, 4e-9, 6e-9).squeeze()


# --------------------------------------------------
# 1. Calculate wind speed and direction
# --------------------------------------------------

wind_speed = np.sqrt(u**2 + v**2)

wind_direction = (
    270 - np.degrees(np.arctan2(v, u))
) % 360

wind = pd.DataFrame({
    "u10": u.values,
    "v10": v.values,
    "wind_speed": wind_speed.values,
    "wind_direction": wind_direction.values
}, index=pd.to_datetime(u.time.values))

wind.index.name = "time"

wind.index = (
    wind.index
    .tz_localize("UTC")
    .tz_convert("Asia/Shanghai")
)

# --------------------------------------------------
# 2. Calculate compass directions BEFORE joining
# --------------------------------------------------

directions = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW"
]

wind["compass_direction"] = pd.cut(
    (wind["wind_direction"] + 11.25) % 360,
    bins=np.arange(0, 360 + 22.5, 22.5),
    labels=directions,
    right=False
)

# --------------------------------------------------
# 3. Prepare concentration and identify NPF
# --------------------------------------------------

threshold = 5000

limit_1 = 2000
limit_2 = 5000

npf = concentration >= threshold

traffic = limit_1 <= concentration <= limit_2

data = pd.DataFrame({
    "concentration": concentration,
    "NPF": npf,
    "traffic": traffic
})

# Join wind and concentration data
data = data.join(wind, how="inner")

# Remove missing values and calm winds
data = data.dropna(
    subset=[
        "concentration",
        "wind_direction",
        "wind_speed",
        "compass_direction"
    ]
)

data = data[data["wind_speed"] > 0.5]

# --------------------------------------------------
# 4. Separate NPF and non-NPF periods
# --------------------------------------------------

npf_data = data[data["NPF"]].copy()
non_npf_data = data[~data["NPF"]].copy()

print("Total observations:", len(data))
print("NPF observations:", len(npf_data))
print("Non-NPF observations:", len(non_npf_data))

# --------------------------------------------------
# 5. Plot wind direction during NPF
# --------------------------------------------------

bins = np.arange(0, 361, 30)

plt.figure(figsize=(9, 5))

plt.hist(
    npf_data["wind_direction"],
    bins=bins,
    edgecolor="black"
)

plt.xlabel("Wind direction (°)")
plt.ylabel("Number of observations")
plt.xticks(bins)
plt.title("Wind directions during NPF")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("wind_NPF.png", bbox_inches="tight", dpi=300)
plt.show()

# --------------------------------------------------
# 6. Plot compass directions during NPF
# --------------------------------------------------

npf_counts = (
    npf_data["compass_direction"]
    .value_counts()
    .reindex(directions, fill_value=0)
)

plt.figure(figsize=(10, 5))

npf_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.xlabel("Wind direction")
plt.ylabel("Number of observations")
plt.title("Wind directions during NPF")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("wind_NPF_compass.png", bbox_inches="tight", dpi=300)
# --------------------------------------------------
# 7. Calculate NPF frequency by wind direction
# --------------------------------------------------

npf_frequency = (
    data.groupby("compass_direction", observed=False)["NPF"]
    .mean() * 100
)

npf_frequency = npf_frequency.reindex(directions)

print("\nNPF frequency by wind direction (%):")
print(npf_frequency.round(2))

# --------------------------------------------------
# 8. Plot NPF frequency by wind direction
# --------------------------------------------------

plt.figure(figsize=(10, 5))

npf_frequency.plot(
    kind="bar",
    edgecolor="black"
)

plt.xlabel("Wind direction")
plt.ylabel("NPF frequency (%)")
plt.title("NPF frequency by wind direction")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "NPF_frequency_by_wind_direction.png",
    bbox_inches="tight",
    dpi=300
)

import numpy as np
import matplotlib.pyplot as plt

# Wind direction data
wind = npf_data["wind_direction"].dropna().to_numpy()

# 30-degree sectors
bins = np.arange(0, 361, 30)

# Calculate frequency in each sector
counts, edges = np.histogram(wind, bins=bins)

# Sector centers
centers = np.deg2rad((edges[:-1] + edges[1:]) / 2)

# Sector width
width = np.deg2rad(30)

# Create polar plot
fig, ax = plt.subplots(
    figsize=(8, 8),
    subplot_kw={"projection": "polar"}
)

# Plot bars
ax.bar(
    centers,
    counts,
    width=width,
    bottom=0,
    color="steelblue",
    edgecolor="black",
    linewidth=0.8,
    alpha=0.85,
    align="center"
)

# Set north at the top
ax.set_theta_zero_location("N")

# Clockwise direction
ax.set_theta_direction(-1)

# Direction labels
ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
ax.set_xticklabels(
    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    fontsize=12
)

# Radial axis
ax.set_ylabel("Number of observations", labelpad=25)
ax.set_rlabel_position(45)

# Grid and appearance
ax.grid(alpha=0.35, linestyle="--")
ax.spines["polar"].set_linewidth(1)

ax.set_title(
    "Wind direction during NPF events",
    fontsize=15,
    fontweight="bold",
    pad=25
)

plt.tight_layout()

plt.savefig(
    "wind_rose_NPF.png",
    dpi=300,
    bbox_inches="tight"
)