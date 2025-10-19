import numpy as np
from scipy.stats import binom
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#1: Simulare pentru a Estima Câștigătorul ---

def simulate_game():
    starter = np.random.choice(['P0', 'P1'])
    n = np.random.randint(1, 7)
    
    if starter == 'P0':
        second_player = 'P1'
        prob_heads = 4/7
    else:
        second_player = 'P0'
        prob_heads = 0.5
        
    m = np.random.binomial(n=(2 * n), p=prob_heads)
    
    winner = starter if n >= m else second_player
    return winner

def run_simulation(num_games=10000):
    wins = {'P0': 0, 'P1': 0}
    for _ in range(num_games):
        wins[simulate_game()] += 1
        
    print("--- Rezultate Simulare (Partea 1) ---")
    print(f"Victorii P0: {wins['P0']} ({(wins['P0']/num_games)*100:.2f}%)")
    print(f"Victorii P1: {wins['P1']} ({(wins['P1']/num_games)*100:.2f}%)")
    print("-" * 35 + "\n")


#2: Definirea Rețelei Bayesiene ---

def define_bayesian_network():
    # Structura rețelei: S->M, N->M, S->W, N->W, M->W
    model = DiscreteBayesianNetwork([
        ('S', 'M'), ('N', 'M'),
        ('S', 'W'), ('N', 'W'), ('M', 'W')
    ])
    
    # Definirea CPD-urilor pentru nodurile rădăcină S (Starter) și N (Die Roll)
    cpd_s = TabularCPD('S', 2, [[0.5], [0.5]], state_names={'S': ['P0', 'P1']})
    cpd_n = TabularCPD('N', 6, [[1/6]]*6, state_names={'N': list(range(1, 7))})

    # CPD pentru M (Heads Count), calculat cu distribuția binomială
    prob_heads_p0, prob_heads_p1 = 0.5, 4/7
    cpd_m_values = []
    for p_heads in [prob_heads_p1, prob_heads_p0]: # Ordinea e importantă: P1 apoi P0
        for n_val in range(1, 7):
            prob_col = [binom.pmf(k=m_val, n=2*n_val, p=p_heads) for m_val in range(13)]
            cpd_m_values.append(prob_col)
    
    cpd_m = TabularCPD('M', 13, np.array(cpd_m_values).T.tolist(),
                       evidence=['S', 'N'], evidence_card=[2, 6],
                       state_names={'M': list(range(13)), 'S': ['P0', 'P1'], 'N': list(range(1, 7))})

    # CPD pentru W (Winner), nod deterministic bazat pe regula n >= m
    cpd_w_values = np.zeros((2, 2 * 6 * 13))
    col_index = 0
    for s_idx in range(2): # 0 pentru P0, 1 pentru P1
        for n_val in range(1, 7):
            for m_val in range(13):
                winner_idx = s_idx if n_val >= m_val else 1 - s_idx
                cpd_w_values[winner_idx, col_index] = 1
                col_index += 1
    
    cpd_w = TabularCPD('W', 2, cpd_w_values.tolist(),
                       evidence=['S', 'N', 'M'], evidence_card=[2, 6, 13],
                       state_names={'W': ['P0', 'P1'], 'S': ['P0', 'P1'], 'N': list(range(1, 7)), 'M': list(range(13))})

    model.add_cpds(cpd_s, cpd_n, cpd_m, cpd_w)
    assert model.check_model()
    
    print("--- Definire Rețea Bayesiană (Partea 2) ---")
    print("Rețeaua Bayesiană a fost definită și validată cu succes.")
    print("-" * 35 + "\n")
    return model


#3: Inferență folosind Modelul ---

def run_inference(model):
    inference_engine = VariableElimination(model)
    
    # Calculăm P(S | M=1) pentru a afla cine a început cel mai probabil
    result = inference_engine.query(variables=['S'], evidence={'M': 1})
    
    print("--- Rezultate Inferență (Partea 3) ---")
    print("Cine a început cel mai probabil, știind că M=1?")
    print(result)
    
    if result.values[1] > result.values[0]:
        print("Concluzie: Este mai probabil ca Jucătorul P1 să fi început jocul.")
    else:
        print("Concluzie: Este mai probabil ca Jucătorul P0 să fi început jocul.")
    print("-" * 35 + "\n")


#Execuția principală 
if __name__ == "__main__":
    run_simulation(10000)
    bayesian_model = define_bayesian_network()
    run_inference(bayesian_model)