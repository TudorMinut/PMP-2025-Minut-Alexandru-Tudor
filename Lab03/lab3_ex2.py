import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([('Rezultat_Zar', 'Bila_Extrasa')])

# CPD pentru nodul rădăcină 'Rezultat_Zar'
cpd_zar = TabularCPD(variable='Rezultat_Zar', variable_card=3,
                     values=[[3/6], [1/6], [2/6]],
                     state_names={'Rezultat_Zar': ['Prim', 'Sase', 'Altul']})

# CPD pentru 'Bila_Extrasa', condiționat de 'Rezultat_Zar'
cpd_bila = TabularCPD(variable='Bila_Extrasa', variable_card=3,
                      values=[[0.3, 0.4, 0.3],  # Probabilități pentru 'Rosu'
                              [0.4, 0.4, 0.5],  # Probabilități pentru 'Albastru'
                              [0.3, 0.2, 0.2]], # Probabilități pentru 'Negru'
                      evidence=['Rezultat_Zar'],
                      evidence_card=[3],
                      state_names={'Bila_Extrasa': ['Rosu', 'Albastru', 'Negru'],
                                   'Rezultat_Zar': ['Prim', 'Sase', 'Altul']})

# Adăugarea CPD-urilor la model și verificarea
model.add_cpds(cpd_zar, cpd_bila)
assert model.check_model()

# Efectuarea inferenței pentru a găsi P(Bila_Extrasa)
inference = VariableElimination(model)
prob_bila_rosie_dist = inference.query(variables=['Bila_Extrasa'])

# Extragerea probabilității specifice pentru bila roșie
prob_rosu_teoretica = prob_bila_rosie_dist.values[0]

print("--- Rezultat Obținut cu Rețeaua Bayesiană (Cod Corectat) ---")
print("Distribuția de probabilitate pentru bila extrasă:")
print(prob_bila_rosie_dist)
print(f"Probabilitatea teoretică de a extrage o bilă roșie este {prob_rosu_teoretica:.6f}")
print(f"Acest rezultat corespunde fracției 19/60.")