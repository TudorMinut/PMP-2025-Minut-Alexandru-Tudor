from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([
    ('O', 'H'),
    ('O', 'W'),
    ('H', 'R'),
    ('W', 'R'),
    ('H', 'E'),
    ('R', 'C')
])

cpd_o = TabularCPD(
    variable='O',
    variable_card=2,
    values=[[0.3], [0.7]],
    state_names={'O': ['cold', 'mild']}
)

cpd_h = TabularCPD(
    variable='H',
    variable_card=2,
    values=[[0.9, 0.2], [0.1, 0.8]],
    evidence=['O'],
    evidence_card=[2],
    state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']}
)

cpd_w = TabularCPD(
    variable='W',
    variable_card=2,
    values=[[0.9, 0.2], [0.1, 0.8]],
    evidence=['O'],
    evidence_card=[2],
    state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']}
)

cpd_r = TabularCPD(
    variable='R',
    variable_card=2,
    values=[[0.6, 0.9, 0.3, 0.5], [0.4, 0.1, 0.7, 0.5]],
    evidence=['H', 'W'],
    evidence_card=[2, 2],
    state_names={'R': ['warm', 'cold'], 'H': ['yes', 'no'], 'W': ['yes', 'no']}
)

cpd_e = TabularCPD(
    variable='E',
    variable_card=2,
    values=[[0.8, 0.2], [0.2, 0.8]],
    evidence=['H'],
    evidence_card=[2],
    state_names={'E': ['high', 'low'], 'H': ['yes', 'no']}
)

cpd_c = TabularCPD(
    variable='C',
    variable_card=2,
    values=[[0.85, 0.4], [0.15, 0.6]],
    evidence=['R'],
    evidence_card=[2],
    state_names={'C': ['comfortable', 'uncomfortable'], 'R': ['warm', 'cold']}
)

model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)

if not model.check_model():
    raise ValueError("Model is not valid.")

#b Probabiliti of computing
inference = VariableElimination(model)

#P(H=yes | C=comfortable)
prob_h_given_c = inference.query(
    variables=['H'],
    evidence={'C': 'comfortable'}
)
print("P(H | C = comfortable):")
print(prob_h_given_c)
print(f"P(H=yes | C=comfortable) is: {prob_h_given_c.values[0]:.4f}")

#P(P=high | C=comfotable)
prob_e_given_c = inference.query(
    variables=['E'],
    evidence={'C': 'comfortable'}
)
print("P(E | C = comfortable):")
print(prob_e_given_c)
print(f"P(E=high | C=comfortable) is: {prob_e_given_c.values[0]:.4f}")


map_hw_given_c = inference.map_query(
    variables=['H', 'W'],
    evidence={'C': 'comfortable'}
)
print("MAP for (H, W) given C = comfortable:")
print(map_hw_given_c)
