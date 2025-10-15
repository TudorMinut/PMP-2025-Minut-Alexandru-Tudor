from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = DiscreteBayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')]) # definim relatiile de dependinte

cpd_s = TabularCPD(variable='S', variable_card=2, 
                   values=[[0.6], [0.4]]) #pentru S=0 avem 0.6 si pentru S=1, 0.4

cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3],  #lista pentru O=0 cu S=0 si S=1
                           [0.1, 0.7]], #lista pentru O=1 cu S=0 si S=1
                   evidence=['S'],
                   evidence_card=[2])   

cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2],  #lista pentru L=0 cu S=0 si S=1
                           [0.3, 0.8]], #lista pentru L=1 cu S=0 si S=1
                   evidence=['S'],
                   evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.8, 0.4, 0.5, 0.1],  #lista pentru M=0 cu S=0, L=0 | S=0, L=1 | S=1, L=0 | S=1, L=1
                           [0.2, 0.6, 0.5, 0.9]], #lista pentru M=1 cu S=0, L=0 | S=0, L=1 | S=1, L=0 | S=1, L=1
                   evidence=['S', 'L'],
                   evidence_card=[2, 2])

model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)
assert model.check_model()

independencies = model.get_independencies()
print(independencies)

inference = VariableElimination(model)

print("{:<30} | {:<25} | {:<15}".format("Atribute (O, L, M)", "P(Spam | Atribute)", "Este Spam?"))

# Iterăm toate combinațiile posibile 
for o_val in [0, 1]:
    for l_val in [0, 1]:
        for m_val in [0, 1]:
            posterior_s = inference.query(variables=['S'], evidence={'O': o_val, 'L': l_val, 'M': m_val})

            # Verificam daca este spam
            prob_spam = posterior_s.values[1]
            classification = "Spam" if prob_spam > 0.5 else "Not Spam"

            # Afișăm rezultatele 
            evidence_str = f"Ofertă={o_val}, Link={l_val}, Lung={m_val}"
            print("{:<30} | {:<25.4f} | {:<15}".format(evidence_str, prob_spam, classification))


pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()
