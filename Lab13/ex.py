import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

dummy_data = np.loadtxt('dummy.csv', delimiter=',')
if dummy_data.shape[1] < 2: 
    dummy_data = np.loadtxt('dummy.csv') 

x = dummy_data[:, 0]
y = dummy_data[:, 1]

x_mean = x.mean()
x_std = x.std()
x_s = (x - x_mean) / x_std

y_mean = y.mean()
y_std = y.std()
y_s = (y - y_mean) / y_std

def fit_poly(x_in, y_in, order, beta_sd):
    x_p = np.vstack([x_in**i for i in range(1, order + 1)]).T
    
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=beta_sd, shape=order)
        sigma = pm.HalfNormal('sigma', sigma=10)
        
        mu = alpha + pm.math.dot(x_p, beta)
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_in)
        
        trace = pm.sample(2000, return_inferencedata=True, progressbar=False)
    return model, trace

order = 5
model_5_10, trace_5_10 = fit_poly(x_s, y_s, order, beta_sd=10)

plt.figure(figsize=(10, 6))
plt.scatter(x, y_s, c='k')
az.plot_hdi(x, trace_5_10.posterior['mu'], color='C0', smooth=False)
post_mean = trace_5_10.posterior['mu'].mean(dim=["chain", "draw"]).values
idx = np.argsort(x)
plt.plot(x[idx], post_mean[idx], 'C0')
plt.title(f'Order {order} Model (sd=10)')
plt.show()

model_5_100, trace_5_100 = fit_poly(x_s, y_s, order, beta_sd=100)
custom_sd = np.array([10, 0.1, 0.1, 0.1, 0.1])
model_5_custom, trace_5_custom = fit_poly(x_s, y_s, order, beta_sd=custom_sd)

plt.figure(figsize=(10, 6))
plt.scatter(x, y_s, c='k')

mu_100 = trace_5_100.posterior['mu'].mean(dim=["chain", "draw"]).values
plt.plot(x[idx], mu_100[idx], label='sd=100', linestyle='--')

mu_custom = trace_5_custom.posterior['mu'].mean(dim=["chain", "draw"]).values
plt.plot(x[idx], mu_custom[idx], label='sd=[10, 0.1, ...]', linestyle='-.')

plt.legend()
plt.title('Comparison of Priors (Order 5)')
plt.show()

np.random.seed(42)
x_500 = np.linspace(x.min(), x.max(), 500)
y_500_true = 2 + 1.5 * x_500 - 0.5 * x_500**2
y_500 = y_500_true + np.random.normal(0, 2, 500)

x_500_s = (x_500 - x_500.mean()) / x_500.std()
y_500_s = (y_500 - y_500.mean()) / y_500.std()

model_500, trace_500 = fit_poly(x_500_s, y_500_s, order=5, beta_sd=10)

plt.figure(figsize=(10, 6))
plt.scatter(x_500, y_500_s, c='k', s=2, alpha=0.3)
az.plot_hdi(x_500, trace_500.posterior['mu'], color='C1', smooth=False)
mu_500 = trace_500.posterior['mu'].mean(dim=["chain", "draw"]).values
plt.plot(x_500, mu_500, 'C1')
plt.title('Order 5 Model with N=500')
plt.show()

orders = [1, 2, 3]
traces = {}
models = {}

for o in orders:
    m, t = fit_poly(x_s, y_s, o, beta_sd=10)
    models[str(o)] = m
    traces[str(o)] = t
    pm.compute_log_likelihood(t, model=m)

comp_waic = az.compare(traces, ic="waic", scale="deviance")
comp_loo = az.compare(traces, ic="loo", scale="deviance")

print(comp_waic)
print(comp_loo)

az.plot_compare(comp_waic)
plt.title("Model Comparison (WAIC)")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x, y_s, c='k')
colors = ['C0', 'C1', 'C2']
for i, o in enumerate(orders):
    mu = traces[str(o)].posterior['mu'].mean(dim=["chain", "draw"]).values
    plt.plot(x[idx], mu[idx], color=colors[i], label=f'Order {o}')
plt.legend()
plt.title("Linear vs Quadratic vs Cubic")
plt.show()
