import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import networkx as nx 

#a) Definire Model HMM
print(" a) Definirea Modelului HMM")
# Spatiul de stari 
states = ["Dificil", "Mediu", "Usor"]
n_states = len(states)

# Spatiul de observatii (ce note pot fi luate)
observations = ["FB", "B", "S", "NS"]
n_observations = len(observations)

# Probabilitati initiale pentru stari
start_probability = np.array([1/3, 1/3, 1/3])

# Probabilitati de tranzitie intre stari
transition_probability = np.array([
    [0.0,  0.5,  0.5 ],  # De la Dificil
    [0.5,  0.25, 0.25],  # De la Mediu
    [0.5,  0.25, 0.25]   # De la Usor
])

# Probabilitati de emisie 
emission_probability = np.array([
    [0.10, 0.20, 0.40, 0.30],  # Probabilitati Dificil
    [0.15, 0.25, 0.50, 0.10],  # Probabilitati Mediu
    [0.20, 0.30, 0.40, 0.10]   # Probabilitati Usor
])

# Crearea si configurarea modelului HMM
model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability
print("Modelul HMM a fost definit ")

#Desenarea diagramei de stări 
print("\nDesenarea diagramei de stări pentru punctul a)")

G = nx.DiGraph()
# Adauga muchiile (tranzitiile) si etichetele lor (probabilitatile)
for i, origin_state in enumerate(states):
    for j, dest_state in enumerate(states):
        prob = transition_probability[i, j]
        if prob > 0: # Adauga o muchie doar daca tranzitia e posibila
            G.add_edge(origin_state, dest_state, weight=prob, label=str(prob))


pos = nx.spring_layout(G, seed=42) 
# Deseneaza graful
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=1.5, arrowstyle='->', arrowsize=20)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Adauga etichetele pe muchii
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Diagrama de Stări a Modelului HMM", size=15)
plt.axis('off') 
plt.show()


# Secventa de observatii (notele)
observed_grades_str = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]
observations_sequence = np.array([0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1]).reshape(-1, 1) #Conversia secventei la indecsi numerici: FB=0, B=1, ...

# b) Probabilitatea de a obtine secvența de observatii
print("\nb) Probabilitatea de a obține secvența de observații")
log_prob_obs = model.score(observations_sequence)
prob_obs = np.exp(log_prob_obs)

print(f"Secvența observată: {observed_grades_str}")
print(f"Probabilitatea obținerii secvenței: {prob_obs:.12f}")

# c) Cea mai probabila secventa de dificultati
print("\nc) Cea mai probabilă secvență de dificultăți")
hidden_states_indices = model.predict(observations_sequence)
hidden_states_labels = [states[i] for i in hidden_states_indices]

print(f"Cea mai probabilă secvență de dificultăți (indecși): {hidden_states_indices}")
print(f"Cea mai probabilă secvență de dificultăți (etichete): {hidden_states_labels}")
