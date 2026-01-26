from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 10
rng = np.random.default_rng(RANDOM_SEED)

def base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

BASE_DIR = base_dir()

def z(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    mu = float(x.mean())
    sd = float(x.std())
    if sd == 0:
        return x * 0.0, mu, sd
    return (x - mu) / sd, mu, sd

def one_hot_season(season_series: pd.Series, baseline: str = "spring"):
    s = season_series.astype(str).str.lower()
    cats = ["spring", "summer", "autumn", "winter"]
    present = [c for c in cats if c in set(s)]
    if not present:
        present = sorted(set(s))
    baseline_used = baseline.lower() if baseline.lower() in present else present[0]

    dummies = pd.get_dummies(s, prefix="season")
    base_col = f"season_{baseline_used}"
    if base_col in dummies.columns:
        dummies = dummies.drop(columns=[base_col])
    cols = sorted(dummies.columns)
    dummies = dummies[cols]
    return dummies, cols

def most_influential_from_posterior(idata, beta_name: str, col_names):
    b = idata.posterior[beta_name].values
    if b.ndim == 2:
        mean_abs = [float(np.mean(np.abs(b.reshape(-1))))]
        return col_names[0], {col_names[0]: mean_abs[0]}
    
    b_reshaped = b.reshape(-1, b.shape[-1])
    mean_abs = np.mean(np.abs(b_reshaped), axis=0)
    influence = {col_names[i]: float(mean_abs[i]) for i in range(len(col_names))}
    idx = int(np.argmax(mean_abs))
    return col_names[idx], influence

def plot_scatter(x, y, xlabel, ylabel, title):
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_posterior_mean_band(x_grid, mu_samples, title, xlabel, ylabel):
    mu_mean = mu_samples.mean(axis=0)
    hdi = az.hdi(mu_samples, hdi_prob=0.94)
    plt.figure()
    plt.plot(x_grid, mu_mean, label='Mean Prediction')
    plt.fill_between(x_grid, hdi[:, 0], hdi[:, 1], alpha=0.3, label='HDI 94%')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    # 1. Incarcarea datelor si explorare vizuala
    csv_path = BASE_DIR / "bike_daily.csv"
    if not csv_path.exists():
        print(f"ATENTIE: Nu am gasit {csv_path}. Asigura-te ca fisierul e in folder.")
        # Generam date sintetice daca fisierul lipseste pentru a putea rula codul
        print("Generare date sintetice pentru demonstratie")
        n = 365
        df = pd.DataFrame({
            'rentals': np.random.poisson(5000, n),
            'temp_c': np.random.normal(15, 10, n),
            'humidity': np.random.beta(2, 2, n),
            'wind_kph': np.abs(np.random.normal(10, 5, n)),
            'is_holiday': np.random.binomial(1, 0.05, n),
            'season': np.random.choice(['spring', 'summer', 'autumn', 'winter'], n)
        })
    else:
        df = pd.read_csv(csv_path)

    needed = ["rentals", "temp_c", "humidity", "wind_kph", "is_holiday", "season"]
    # Verificam daca coloanele exista
    available = [c for c in needed if c in df.columns]
    df = df[available].dropna().copy()
    
    print("Date incarcate. Dimensiune:", df.shape)
    print(df.head())

    # Generam grafice
    plot_scatter(df["temp_c"], df["rentals"], "temp_c", "rentals", "Rentals vs temp_c")
    plot_scatter(df["humidity"], df["rentals"], "humidity", "rentals", "Rentals vs humidity")
    plot_scatter(df["wind_kph"], df["rentals"], "wind_kph", "rentals", "Rentals vs wind_kph")

    # 2. Standardizare si pregatirea datelor
    y = df["rentals"].to_numpy(float)
    y_z, y_mu, y_sd = z(y)

    temp = df["temp_c"].to_numpy(float)
    hum = df["humidity"].to_numpy(float)
    wind = df["wind_kph"].to_numpy(float)

    temp_z, temp_mu, temp_sd = z(temp)
    hum_z, hum_mu, hum_sd = z(hum)
    wind_z, wind_mu, wind_sd = z(wind)

    is_holiday = df["is_holiday"].to_numpy(float)
    season_dummies, season_cols = one_hot_season(df["season"], baseline="spring")

    # Matrice pentru modelul liniar
    X_lin = np.column_stack([temp_z, hum_z, wind_z, is_holiday, season_dummies.to_numpy(float)])
    colnames_lin = ["temp_c_z", "humidity_z", "wind_kph_z", "is_holiday"] + season_cols

    # Matrice pentru modelul polinomial
    temp2 = temp_z ** 2
    temp2 = temp2 - temp2.mean()
    X_poly = np.column_stack([temp_z, temp2, hum_z, wind_z, is_holiday, season_dummies.to_numpy(float)])
    colnames_poly = ["temp_c_z", "temp_c_z2_centered", "humidity_z", "wind_kph_z", "is_holiday"] + season_cols

    # Model Liniar
    print("Rulare Model Liniar")
    with pm.Model() as m_lin:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        betas = pm.Normal("betas", mu=0.0, sigma=1.0, shape=X_lin.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = alpha + pm.math.dot(X_lin, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_z)

        id_lin = pm.sample(draws=1500, tune=1500, chains=2, target_accept=0.9,
                           random_seed=RANDOM_SEED, progressbar=True)
        #Calculam log likelihood explicit pentru WAIC
        pm.compute_log_likelihood(id_lin)

    # Model Polinomial
    print("Rulare Model Polinomial")
    with pm.Model() as m_poly:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        betas = pm.Normal("betas", mu=0.0, sigma=1.0, shape=X_poly.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = alpha + pm.math.dot(X_poly, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_z)

        id_poly = pm.sample(draws=1500, tune=1500, chains=2, target_accept=0.9,
                            random_seed=RANDOM_SEED, progressbar=True)
        #Calculam log likelihood explicit pentru WAIC
        pm.compute_log_likelihood(id_poly)

    # 3. Inferenta si diagnostic
    print("Sumar Model Liniar (95% HDI):")
    print(az.summary(id_lin, var_names=["alpha", "betas", "sigma"], hdi_prob=0.95))

    print("Sumar Model Polinomial (95% HDI):")
    print(az.summary(id_poly, var_names=["alpha", "betas", "sigma"], hdi_prob=0.95))

    top_lin, infl_lin = most_influential_from_posterior(id_lin, "betas", colnames_lin)
    top_poly, infl_poly = most_influential_from_posterior(id_poly, "betas", colnames_poly)

    print(f"Cea mai influenta variabila (liniar): {top_lin}")
    print(f"Cea mai influenta variabila (polinomial): {top_poly}")

    # 4. Comparare modele (WAIC) si PPC
    # Acum id_lin si id_poly contin grupul log_likelihood
    waic_lin = az.waic(id_lin)
    waic_poly = az.waic(id_poly)

    print("WAIC liniar:", waic_lin.elpd_waic)
    print("WAIC polinomial:", waic_poly.elpd_waic)

    # Vizualizare PPC pentru modelul polinomial
    temp_grid = np.linspace(temp.min(), temp.max(), 120)
    temp_grid_z = (temp_grid - temp_mu) / (temp_sd if temp_sd != 0 else 1.0)
    temp_grid_z2 = temp_grid_z ** 2
    temp_grid_z2 = temp_grid_z2 - temp2.mean()

    hum0 = 0.0
    wind0 = 0.0
    holiday0 = 0.0
    season0 = np.zeros(len(season_cols))

    Xg_poly = np.column_stack([
        temp_grid_z,
        temp_grid_z2,
        np.full_like(temp_grid_z, hum0),
        np.full_like(temp_grid_z, wind0),
        np.full_like(temp_grid_z, holiday0),
        np.tile(season0, (len(temp_grid_z), 1))
    ])

    def posterior_mu_samples(idata, Xg):
        post = idata.posterior
        a = post["alpha"].values.reshape(-1)
        b = post["betas"].values.reshape(-1, Xg.shape[1])
        mu_z = a[:, None] + b @ Xg.T
        return y_mu + y_sd * mu_z

    mu_poly = posterior_mu_samples(id_poly, Xg_poly)

    plot_posterior_mean_band(temp_grid, mu_poly,
                             "PPC: Predictie Rentals vs Temperatura (Polinomial)",
                             "Temperatura (C)", "Rentals Estimati")

    # 5. Constructie target binar (High Demand)
    Q = float(np.percentile(y, 75))
    is_high = (y >= Q).astype(int)
    print("Pragul Q (75%):", Q)
    print("Proportie High Demand:", is_high.mean())

    # 6. Regresie Logistica
    print("Rulare Regresie Logistica")
    with pm.Model() as m_log:
        alpha = pm.Normal("alpha", mu=0.0, sigma=2.5)
        betas = pm.Normal("betas", mu=0.0, sigma=2.5, shape=X_poly.shape[1])
        
        logits = alpha + pm.math.dot(X_poly, betas)
        p = pm.Deterministic("p", pm.math.sigmoid(logits))
        
        pm.Bernoulli("y_obs", p=p, observed=is_high)

        id_log = pm.sample(draws=2000, tune=2000, chains=2, target_accept=0.9,
                           random_seed=RANDOM_SEED, progressbar=True)
        # calculam si aici daca vom folosi WAIC pe viitor
        pm.compute_log_likelihood(id_log)

    # 7. Analiza deciziei
    summary_log = az.summary(id_log, var_names=["alpha", "betas"], hdi_prob=0.95)
    print(summary_log)

    top_log, infl_log = most_influential_from_posterior(id_log, "betas", colnames_poly)
    print(f"Cea mai influenta variabila in modelul logistic: {top_log}")
    
    # Vizualizare curba de probabilitate
    post = id_log.posterior
    a = post["alpha"].values.reshape(-1)
    b = post["betas"].values.reshape(-1, Xg_poly.shape[1])
    logits = a[:, None] + b @ Xg_poly.T
    p_samples = 1.0 / (1.0 + np.exp(-logits))

    p_mean = p_samples.mean(axis=0)
    p_hdi = az.hdi(p_samples, hdi_prob=0.94)

    plt.figure()
    plt.plot(temp_grid, p_mean, label="Probabilitate medie")
    plt.fill_between(temp_grid, p_hdi[:, 0], p_hdi[:, 1], alpha=0.3, label="HDI 94%")
    plt.axhline(0.5, linestyle="--", color='gray')
    plt.title("Probabilitatea de High Demand vs Temperatura")
    plt.xlabel("Temperatura (C)")
    plt.ylabel("P(High Demand)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
