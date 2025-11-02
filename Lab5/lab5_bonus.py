import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import networkx as nx 

# a) Definire Model HMM
print("a) Definirea Modelului HMM")
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
print("Modelul HMM a fost definit")

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
log_prob_seq, hidden_states_indices = model.decode(observations_sequence)
hidden_states_labels = [states[i] for i in hidden_states_indices]
prob_seq = np.exp(log_prob_seq)

print(f"Cea mai probabilă secvență de dificultăți (etichete): {hidden_states_labels}")
print(f"Probabilitatea acestei secvențe: {prob_seq:.12e}")



###
###
# Bonus:
###
###

def viterbi_manual(obs_seq, states, start_p, trans_p, emit_p):
 
    # Tabela Viterbi care stochează probabilitățile la fiecare pas
    viterbi_table = [{}]
    # Tabela care stochează calea optimă (backpointers)
    path = {}

    # Inițializarea (pentru prima observație, t = 0)
    for state in states:
        # Calculăm probabilitatea de a fi în 'state' și de a observa primul element
        viterbi_table[0][state] = start_p[state] * emit_p[state][obs_seq[0]]
        path[state] = [state]

    # Recursivitatea (pentru restul observațiilor, t > 0)
    for t in range(1, len(obs_seq)):
        viterbi_table.append({})
        new_path = {}

        for current_state in states:
            # Găsim probabilitatea maximă și starea anterioară care duce la starea curentă
            (prob, prev_state) = max(
                (viterbi_table[t-1][prev_st] * trans_p[prev_st][current_state] * emit_p[current_state][obs_seq[t]], prev_st)
                for prev_st in states
            )
            
            # Stocăm probabilitatea maximă în tabelă
            viterbi_table[t][current_state] = prob
            # Adăugăm starea curentă la calea optimă găsită până acum
            new_path[current_state] = path[prev_state] + [current_state]

        # Actualizăm calea cu noile căi optime
        path = new_path

    # Terminarea
    # La finalul secvenței, găsim probabilitatea maximă și ultima stare
    (prob, final_state) = max((viterbi_table[len(obs_seq)-1][state], state) for state in states)

    # Returnăm probabilitatea finală și calea corespunzătoare stării finale
    return (prob, path[final_state])

# Definirea parametrilor HMM pentru implementarea manuală
# Folosim dicționare pentru o citire mai ușoară, conform enunțului problemei.
states_bonus = ['Dificil', 'Mediu', 'Ușor']
observation_sequence_bonus = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]

start_probability_bonus = {
    'Dificil': 1/3, 
    'Mediu': 1/3, 
    'Ușor': 1/3
}

transition_probability_bonus = {
    'Dificil': {'Dificil': 0.0, 'Mediu': 0.5, 'Ușor': 0.5},
    'Mediu':   {'Dificil': 0.5, 'Mediu': 0.25, 'Ușor': 0.25},
    'Ușor':    {'Dificil': 0.5, 'Mediu': 0.25, 'Ușor': 0.25}
}

emission_probability_bonus = {
    'Dificil': {'FB': 0.1, 'B': 0.2, 'S': 0.4, 'NS': 0.3},
    'Mediu':   {'FB': 0.15, 'B': 0.25, 'S': 0.5, 'NS': 0.1},
    'Ușor':    {'FB': 0.2, 'B': 0.3, 'S': 0.4, 'NS': 0.1}
}

# --- Apelarea funcției Viterbi manuale și afișarea rezultatului 
final_prob, most_probable_sequence = viterbi_manual(
    observation_sequence_bonus,
    states_bonus,
    start_probability_bonus,
    transition_probability_bonus,
    emission_probability_bonus
)

print(f"\nCea mai probabilă secvență de dificultăți (implementare manuală): {most_probable_sequence}")
print(f"Probabilitatea acestei secvențe (implementare manuală): {final_prob:.12e}")
