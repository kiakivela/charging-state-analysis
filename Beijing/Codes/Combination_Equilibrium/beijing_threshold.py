import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import aerosol.functions as af

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

start = "2020-10-01 00:00:00"
end = "2022-06-29 23:59:59"

neg_i = neg_i.loc[start:end]
neg_p = neg_p.loc[start:end]

neg_p = neg_p[
    (neg_p.index.hour >= 23) |
    (neg_p.index.hour < 6)
]

neg_i = neg_i[
    (neg_i.index.hour >= 23) |
    (neg_i.index.hour < 6)
]

size_ranges = [
    (3e-9, 4e-9),
    (4e-9, 6e-9)
]

diameters_nm = neg_p.columns.to_numpy()

nRangesN = []

for lower, upper in size_ranges:

    n = af.calc_conc(
        neg_p, lower, upper
    )

    nRangesN.append(n.iloc[:, 0].to_numpy())

nRangesN = np.column_stack(nRangesN)


print("nRangesN shape:", nRangesN.shape)
print("finite:", np.isfinite(nRangesN).sum(axis=0))

prcRangesN = np.nanpercentile(
    nRangesN,
    np.arange(0, 101),
    axis=0
)

n_ranges = len(size_ranges)

slopes = np.full((91, n_ranges), np.nan)

for i in range(91):

    x = np.arange(i, i + 11)

    for j in range(n_ranges):

        y_raw = prcRangesN[i:i + 11, j]

        if np.all(y_raw > 0) and np.all(np.isfinite(y_raw)):

            y = np.log10(y_raw)

            slope, intercept = np.polyfit(x, y, 1)

            slopes[i, j] = slope

thrshld = np.empty(n_ranges)
trshInd = np.empty(n_ranges, dtype=int)
limitInd = np.empty(n_ranges, dtype=int)

for i in range(n_ranges):
    dSlopes = np.diff(slopes[:, i])

    dSloPos = np.concatenate([
        [False],
        (dSlopes[1:] >= 0) & (dSlopes[:-1] < 0),
        [False]
    ])

    minimSlopes = slopes[dSloPos, i]

    dSloNeg = np.concatenate([
        [True],
        (dSlopes[1:] < 0) & (dSlopes[:-1] >= 0),
        [True]
    ])

    maximSlopes = slopes[dSloNeg, i]

    smallerValid = np.zeros(len(maximSlopes))
    largerValid = np.zeros(len(maximSlopes))

    for j in range(1, len(maximSlopes) - 1):

        smallerValid[j] = np.sum(
            minimSlopes[:j] < maximSlopes[j]
        )

        largerValid[j] = np.sum(
            minimSlopes[j:] < maximSlopes[j]
        )

    latter_candidates = np.where(largerValid == 1)[0]

    former_candidates = np.where(smallerValid == 1)[0]

    if len(latter_candidates) == 0:
        latterMax = None
    else:
        latterMax = latter_candidates[0]

    if len(former_candidates) == 0:
        formerMax = None
    else:
        formerMax = former_candidates[-1]

    if latterMax is not None and formerMax is not None:

        if maximSlopes[latterMax] > maximSlopes[formerMax]:
            limitInd[i] = latterMax
            use_minimum = False
        else:
            limitInd[i] = formerMax
            use_minimum = False

    elif latterMax is not None:
        limitInd[i] = latterMax
        use_minimum = False

    elif formerMax is not None:
        limitInd[i] = formerMax
        use_minimum = False

    else:
        use_minimum = True

    if use_minimum:

        # Find the minimum slope
        min_index = np.nanargmin(slopes[:, i])
        slopeTrsh = slopes[min_index, i]

        # Find where the slope starts increasing above the minimum
        candidates = np.where(
            slopes[min_index + 1:, i] > slopeTrsh
        )[0]

        if len(candidates) == 0:
            raise ValueError(
                f"No threshold crossing found for minimum "
                f"of column {i}"
            )

        trshInd[i] = candidates[0] + min_index + 1

    else:

        dSloNegInd = np.flatnonzero(dSloNeg)

        max_index = dSloNegInd[int(limitInd[i])]

        slopeTrsh = slopes[max_index, i]

        start_index = max_index + 1

        candidates = np.where(
            slopes[start_index:, i] > slopeTrsh
        )[0]

        if len(candidates) == 0:
            raise ValueError(
                f"No threshold crossing found for column {i}"
            )

        trshInd[i] = candidates[0] + start_index

    thrshld[i] = prcRangesN[
        trshInd[i] + 10,
        i
    ]

print(thrshld)

percentiles = np.nanpercentile(
    nRangesN,
    np.arange(0, 101),
    axis=0
)

print(np.isfinite(percentiles).sum(axis=0))

plt.figure()

for i, (lower, upper) in enumerate(size_ranges):
    plt.semilogy(
        percentiles[:, i],
        label=f"{lower*1e9:.0f}–{upper*1e9:.0f} nm"
    )

plt.xlabel("Percentile")
plt.ylabel("Concentration")
plt.legend()
plt.show()

plt.figure()
for i, (lower, upper) in enumerate(size_ranges):
    plt.plot(
        slopes[:, i],
        label=f"{lower*1e9:.0f}–{upper*1e9:.0f} nm"
    )
plt.xlabel("Window")
plt.ylabel("Slope")
plt.legend()
plt.show()