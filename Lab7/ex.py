import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Datele problemei: 10 observații ale nivelului de zgomot în dB
data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

if __name__ == '__main__':
    # Calculul mediei datelor, care va fi folosită la subpunctul a)
    x_bar = data.mean()
    print(f"Sample mean = {x_bar:.2f}, Sample std = {data.std(ddof=1):.2f}")

    # a) Definirea modelului Bayesian în PyMC cu priori slabi.
    #    Alegem x = media datelor (x_bar) pentru priorul lui mu.
    # b) Inferența asupra lui μ și σ și calculul HDI de 95%.
    with pm.Model() as weak_model:
        # a) Definirea priorilor slab informativi
        mu = pm.Normal("mu", mu=x_bar, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # a) Definirea verosimilității (likelihood)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
        
        # b) Rularea inferenței pentru a obține distribuțiile a posteriori
        trace_weak = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
        # b) Generarea sumarului, care include media, deviația standard și HDI de 95%
        summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

    print("\nPosterior summaries (Weak Prior) - Rezultate pentru subpunctul b):")
    print(summary_weak)

    # c)
    # Comparația estimărilor Bayesiene (din modelul cu priori slabi)
    # cu estimările frequentiste (media și deviația standard a eșantionului).
    print("\nComparație cu estimările frequentiste - Subpunctul c):")
    print(f"Mean (frequentist): {np.mean(data):.2f}")
    print(f"SD (frequentist):   {np.std(data, ddof=1):.2f}")
    print("Discuție: După cum se observă, estimările a posteriori sunt aproape identice cu cele frequentiste, deoarece priorii slabi au permis datelor să domine rezultatul.")


    # d)
    # Investigarea efectului unui prior puternic. Definim un nou model cu
    # un prior informativ pentru medie, centrat pe o convingere că μ este 50.
    with pm.Model() as strong_model:
        # d) Prior puternic pentru medie
        mu = pm.Normal("mu", mu=50, sigma=1)
        # Același prior slab pentru deviația standard
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # Definirea verosimilității (likelihood)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
        
        # d) Rularea inferenței pentru noul model
        trace_strong = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)
        summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)

    print("\nPosterior summaries (Strong Prior) - Rezultate pentru subpunctul d):")
    print(summary_strong)
    #print("Media a posteriori pentru 'mu' a fost 'trasă' de la 58.0 (media datelor) 
    #spre 50 (media priorului), demonstrând influența unui prior puternic și informativ.")


    az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Weak Prior (subpunctele a, b)", fontsize=14)
    plt.show()

    az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Strong Prior (subpunctul d)", fontsize=14)
    plt.show()
