import pymc as pm
import arviz as az
import numpy as np

def main():
    # Datele din problema: publicitate (x) si vanzari (y)
    x_val = [1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]
    y_val = [5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0]

    with pm.Model() as model:
        # a) Estimarea coeficientilor
        # Definim "priors" (ipotezele initiale). Nu stim valorile, deci punem deviatii mari.
        alpha = pm.Normal("alpha", 0, 10)      # Intercept
        beta = pm.Normal("beta", 0, 10)        # Panta (Slope)
        sigma = pm.HalfNormal("sigma", 10)     # Zgomotul (Eroarea)
        
        # Folosim pm.Data pentru a putea schimba datele mai tarziu (la punctul c)
        # Acestea sunt variabile "shared" care pot fi actualizate
        x_shared = pm.Data("x", x_val)
        y_shared = pm.Data("y", y_val)
        
        # Definim relatia liniara si legatura cu datele observate
        # Important: 'shape' asigura ca dimensiunile se potrivesc cand schimbam datele
        pm.Normal("y_obs", mu=alpha + beta * x_shared, sigma=sigma, observed=y_shared, shape=x_shared.shape)
        
        # Rulam algoritmul MCMC pentru a gasi parametrii (antrenarea modelului)
        trace = pm.sample(2000, chains=2, random_seed=42, progressbar=False)

    # a) si b) Afisam coeficientii estimati (Mean) si intervalele de credibilitate (HDI)
    print("\nCoeficienti si HDI (95%)")
    print(az.summary(trace, var_names=["alpha", "beta"], hdi_prob=0.95)[["mean", "hdi_2.5%", "hdi_97.5%"]])

    # c) Predictii pentru noile valori de publicitate
    new_x = [12.0, 13.0]
    # Cream un Y fals de aceeasi lungime cu new_x (2 elemente) pentru a satisface cerinta de shape a modelului
    # Valorile din dummy_y sunt ignorate la predictie, conteaza doar lungimea vectorului
    dummy_y = [0, 0] 

    with model:
        # Inlocuim datele originale (20 puncte) cu noile date (2 puncte)
        pm.set_data({"x": new_x, "y": dummy_y})
        # Generam predictiile folosind distributiile invatate anterior
        ppc = pm.sample_posterior_predictive(trace, var_names=["y_obs"], progressbar=False)
    
    # Calculam media si intervalele pentru predictii
    means = ppc.posterior_predictive["y_obs"].mean(dim=["chain", "draw"])
    hdis = az.hdi(ppc.posterior_predictive, hdi_prob=0.95)["y_obs"]

    print("\nPredictii Vanzari (Mean & HDI)")
    for val, mean, hdi in zip(new_x, means.values, hdis.values):
        print(f"Cheltuieli {val}k: Predictie {mean:.2f}k | Interval [{hdi[0]:.2f}, {hdi[1]:.2f}]")

if __name__ == "__main__":
    main()
