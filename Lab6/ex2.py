import numpy as np
from scipy.stats import gamma
import arviz as az
import matplotlib.pyplot as plt

total_calls = 180
total_hours = 10

alpha_prior = 1.0
beta_prior = 0.001

alpha_posterior = alpha_prior + total_calls
beta_posterior = beta_prior + total_hours

print(f"Distribuția posterioară a lui λ este Gamma(α'={alpha_posterior}, β'={beta_posterior:.3f})\n")

posterior_distribution = gamma(a=alpha_posterior, scale=1/beta_posterior)

hdi_interval = posterior_distribution.interval(0.94)
mode_lambda = (alpha_posterior - 1) / beta_posterior

print(f"Intervalul HDI de 94% calculat analitic este: [{hdi_interval[0]:.2f}, {hdi_interval[1]:.2f}]")
print(f"Cea mai probabilă valoare (modul) este: {mode_lambda:.2f}\n")


posterior_samples = posterior_distribution.rvs(size=10000)


az.plot_posterior(
    {'λ (rata de apeluri)': posterior_samples},
    hdi_prob=0.94
)

plt.show()
