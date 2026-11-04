import numpy as np

nt = np.array([500, 5000, 10000, 20000], dtype=float)  
N  = np.array([6021, 5972, 5915, 5860], dtype=float)   

p, cov = np.polyfit(nt, N, 1, cov=True)
m, b = p

m_err = np.sqrt(cov[0, 0])
b_err = np.sqrt(cov[1, 1])

print(f"Slope: {m:.6e} ± {m_err:.6e}")
print(f"Intercept: {b:.2f} ± {b_err:.2f}")

tau = -m / b
tau_err = tau * np.sqrt((m_err/m)**2 + (b_err/b)**2)

print(f"Tau: {tau:.3e} ± {tau_err:.3e}")

