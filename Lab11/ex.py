import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Citim fisierul CSV local
try:
    df = pd.read_csv("Prices.csv")
    print(f"Date incarcate cu succes: {len(df)} inregistrari.")
except FileNotFoundError:
    print("Eroare: Fisierul 'Prices.csv' nu a fost gasit.")
    print("Te rog creeaza fisierul CSV inainte de a rula scriptul.")
    exit()

# Definim variabilele pentru regresie
# y: Pretul (variabila dependenta)
price = df['Price'].values

# x1: Frecventa procesorului
speed = df['Speed'].values
# x2: Logaritm natural din dimensiunea Hard Disk-ului
hard_drive_log = np.log(df['HardDrive'].values)

# a) Definirea modelului in PyMC
print("\n--- a) Construirea modelului si esantionarea ---")
with pm.Model() as model:
    # Priori slab informativi pentru coeficienti
    alpha = pm.Normal('alpha', mu=0, sigma=1000)
    beta1 = pm.Normal('beta1', mu=0, sigma=100)
    beta2 = pm.Normal('beta2', mu=0, sigma=100)
    
    # Sigma trebuie sa fie pozitiv (HalfNormal)
    sigma = pm.HalfNormal('sigma', sigma=100)

    # Relatia liniara: mu = alpha + beta1*x1 + beta2*x2
    mu = alpha + beta1 * speed + beta2 * hard_drive_log

    # Likelihood (distributia datelor observate)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=price)

    # Generarea esantioanelor (MCMC)
    idata = pm.sample(draws=2000, tune=1000, return_inferencedata=True, random_seed=42)

# b) & c) Analiza parametrilor
print("\n--- b) Estimari 95% HDI ---")
# Afisam media, deviatia standard si intervalele HDI
summary = az.summary(idata, var_names=['alpha', 'beta1', 'beta2', 'sigma'], hdi_prob=0.95)
print(summary[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%']])

print("\n--- c) Utilitate predictori ---")
# Calculam intervalele HDI pentru beta1 si beta2
hdi_beta1 = az.hdi(idata.posterior['beta1'], hdi_prob=0.95)
hdi_beta2 = az.hdi(idata.posterior['beta2'], hdi_prob=0.95)

# Extragem valorile intervalelor
b1_low, b1_high = hdi_beta1.x.values
b2_low, b2_high = hdi_beta2.x.values

print(f"HDI Beta1 (Speed): [{b1_low:.2f}, {b1_high:.2f}]")
print(f"HDI Beta2 (logHD): [{b2_low:.2f}, {b2_high:.2f}]")

# Daca 0 nu este in interval, predictorul este semnificativ
if (b1_low > 0 or b1_high < 0) and (b2_low > 0 or b2_high < 0):
    print("Concluzie: Ambii predictori sunt utili (0 nu este in interval).")
else:
    print("Concluzie: Unul dintre predictori ar putea sa nu fie util.")

# d) & e) Predictie pentru valori specifice
print("\n--- d) & e) Predictie pentru Speed=33, HD=540 ---")

new_speed = 33
new_hd_log = np.log(540)

# Extragem lanturile de esantionare
post = idata.posterior
alpha_s = post['alpha'].values.flatten()
beta1_s = post['beta1'].values.flatten()
beta2_s = post['beta2'].values.flatten()
sigma_s = post['sigma'].values.flatten()

# Calculam media asteptata (mu) pentru noile valori
mu_vals = alpha_s + beta1_s * new_speed + beta2_s * new_hd_log

# d) Interval HDI 90% pentru pretul MEDIU
hdi_mu = az.hdi(mu_vals, hdi_prob=0.90)
print(f"d) 90% HDI pret mediu asteptat: [{hdi_mu[0]:.2f}, {hdi_mu[1]:.2f}]")

# e) Interval HDI 90% pentru o predictie INDIVIDUALA (adaugam zgomot sigma)
y_pred_vals = np.random.normal(mu_vals, sigma_s)
hdi_pred = az.hdi(y_pred_vals, hdi_prob=0.90)
print(f"e) 90% HDI predictie individuala: [{hdi_pred[0]:.2f}, {hdi_pred[1]:.2f}]")
