import numpy as np
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from itertools import product

# a)
model = MarkovNetwork([('A1', 'A2'), ('A1', 'A3'), ('A2', 'A4'), 
                         ('A2', 'A5'), ('A3', 'A4'), ('A4', 'A5')])

cliques = model.get_cliques()
print("a) Clicile maximale ale modelului sunt:")
print(cliques)
print("-" * 40)

# b)
def state_to_value(state):
    return -1 if state == 0 else 1

factors = []
for clique in cliques:
    cardinality = [2] * len(clique)
    potential_values = np.zeros(np.prod(cardinality))
    
    for i, state_combination in enumerate(product(range(2), repeat=len(clique))):
        values = [state_to_value(s) for s in state_combination]
        potential_values[i] = np.exp(np.sum(values))
        
    factor = DiscreteFactor(clique, cardinality, potential_values)
    factors.append(factor)

model.add_factors(*factors)

inference = VariableElimination(model)
map_state_result = inference.map_query(variables=['A1', 'A2', 'A3', 'A4', 'A5'])
best_configuration_values = {var: state_to_value(state) for var, state in map_state_result.items()}

print("\nb) Cea mai buna configuratie (starea de probabilitate maxima) este:")
print(best_configuration_values)
