import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def main():
    y_vals = [0, 5, 10]    # Numarul de cumparatori observati
    theta_vals = [0.2, 0.5] # Probabilitatea de a cumpara

    fig_a, axes_a = plt.subplots(3, 2, figsize=(12, 10))
    fig_a.suptitle("a) Distributia posterioara pentru n (nr. total vizitatori)")

    fig_c, axes_c = plt.subplots(3, 2, figsize=(12, 10))
    fig_c.suptitle("c) Posterior Predictive pentru Y* (viitori cumparatori)")

    print("Se ruleaza inferenta pentru cele 6 scenarii...")

    # Iteram prin fiecare combinatie de Y si theta
    for i, y_obs in enumerate(y_vals):
        for j, theta in enumerate(theta_vals):

            with pm.Model() as model:
                n = pm.Poisson("n", mu=10)
                
                y = pm.Binomial("y", n=n, p=theta, observed=y_obs)

                # Inference
                idata = pm.sample(1000, tune=1000, chains=1, cores=1, progressbar=False)

                # Posterior Predictive
                pm.sample_posterior_predictive(
                    idata, extend_inferencedata=True, progressbar=False
                )

            #Posteriorul parametrului n
            az.plot_posterior(idata, var_names=["n"], ax=axes_a[i, j])
            axes_a[i, j].set_title(f"Y_obs={y_obs}, theta={theta}")

            #Posterior Predictive pentru datele Y*
            az.plot_dist(idata.posterior_predictive["y"], ax=axes_c[i, j], color="C1")
            axes_c[i, j].set_title(f"Y_obs={y_obs}, theta={theta}")
            axes_c[i, j].set_xlabel("Y* (Cumparatori viitori)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


"""
b)1. Efectul lui Y:
   - Cand Y creste, distributia posterioara a lui n se deplaseaza spre dreapta (media lui n creste).
   - Explicatie: n reprezinta numarul total de vizitatori. Astfel, n trebuie sa fie intotdeauna mai mare sau egal cu Y. 
     Daca observam mai multi cumparatori, este necesar sa fi fost mai multi vizitatori in magazin.

2. Efectul lui theta:
   - Exista o relatie inversa intre theta si n.
   - Cand theta este mic (0.2): Pentru a obtine acelasi numar de vanzari (ex: Y=5), modelul estimeaza un n mult mai mare. 
     (Daca doar 20% cumpara, ai nevoie de multi vizitatori ca sa ajungi la 5 vanzari).
   - Cand theta este mare (0.5): Distributia lui n este mai apropiata de valoarea lui Y. 
     (Daca 50% cumpara, nu ai nevoie de mult mai multi vizitatori decat cei care au cumparat deja).


d)1. - Posteriorul pentru n: Este o distributie de probabilitate pentru un parametru necunoscut. 
     Ne spune "Cati oameni au fost in magazin astazi?".
   - Predictive Posterior (Y*): Este o distributie de probabilitate pentru date viitoare (observabile). 
     Ne spune "Cati oameni ne asteptam sa cumpere maine?".

2. - Distributia predictiva (Y*) este, de regula, mai larga (are o varianta mai mare) decat posteriorul lui n.
   - Motivul: Posteriorul predictiva insumeaza doua surse de incertitudine:
     a) Incertitudinea epistemica asupra lui n (nu stim sigur cati vizitatori sunt).
     b) Aleatoriul inerent procesului Binomial (chiar daca am sti exact n, nu stim sigur cine va cumpara - datul cu banul).
"""
