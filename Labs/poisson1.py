import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Load your time-series count data
# Example: one count value per equal time interval
counts = np.loadtxt("/home/kivekia/Downloads/measurement1_hist.txt", skiprows=1)

# Expectation value (lambda)
lam = np.mean(counts)

print(f"Estimated Poisson expectation value λ = {lam:.3f}")
print(f"Standard deviation (Poisson) = {np.sqrt(lam):.3f}")

# ---------------------------
# Plot histogram of counts
# ---------------------------
bins = np.arange(counts.min(), counts.max() + 2) - 0.5

plt.figure()
plt.hist(counts, bins=bins, density=True, alpha=0.6, label="Measured data")

# Plot Poisson model
x = np.arange(counts.min(), counts.max() + 1)
plt.plot(x, poisson.pmf(x, lam), 'o-', label=f"Poisson fit (λ={lam:.2f})")

plt.xlabel("Counts per interval")
plt.ylabel("Probability")
plt.title("Poisson Fit to Radiation Counts")
plt.legend()
plt.show()

lambda_est=np.sum(counts[:,0]*counts[:,1]) / sum(counts[:, 1])
print(lambda_est)

channels = np.arange(len(counts))  # channel number as pulse height proxy

min_len = min(len(channels), len(counts))

channels=channels[:min_len]
counts=counts[:min_len]

mean_channel = np.sum(channels * counts) / np.sum(counts)

variance = np.sum(counts * (channels - mean_channel)**2) / np.sum(counts)
std_dev = np.sqrt(variance)

print(f"Mean channel ⟨H⟩ = {mean_channel:.3f}")
print(f"Standard deviation σ = {std_dev:.3f}")

