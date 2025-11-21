import numpy as np
from hmmlearn import hmm

# a) Defining Modelului HMM
print("a) Definirea Modelului HMM")

# State space
states = ["L", "M", "H"]
n_states = len(states)

# Observation space
observations = ["W", "R", "S"]
n_observations = len(observations)

# Start probabilities
start_probability = np.array([0.4, 0.3, 0.3])

# Transition probability matrix P(state_t | state_{t-1})
transition_probability = np.array([
    [0.6, 0.3, 0.1],  # From L to [L, M, H]
    [0.2, 0.7, 0.1],  # From M to [L, M, H]
    [0.3, 0.2, 0.5]   # From H to [L, M, H]
])

# Emission probability matrix P(observation | state)
emission_probability = np.array([
    [0.7, 0.2, 0.1],  # Given L, P of [W, R, S]
    [0.1, 0.7, 0.2],  # Given M, P of [W, R, S]
    [0.1, 0.2, 0.7]   # Given H, P of [W, R, S]
])

model = hmm.CategoricalHMM(n_components=n_states, n_iter=100)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability
print("Modelul HMM a fost definit.\n")


# b) Computing Probability of an Observation Sequence using the Forward Algorithm
print("b) Computing Theoretical Probability (Forward Algorithm)")
obs_seq_str = ["W", "R", "S"]
print(f"Observation Sequence: {obs_seq_str}")

obs_map = {obs: i for i, obs in enumerate(observations)}
obs_seq_int = np.array([obs_map[obs] for obs in obs_seq_str]).reshape(-1, 1)

log_probability = model.score(obs_seq_int)
prob_theoretical = np.exp(log_probability) # Storing result in prob_theoretical

print(f"Log probability of the sequence: {log_probability:.4f}")
print(f"Probability of the sequence P(W, R, S): {prob_theoretical:.6f}\n")


# c) Finding the Most Likely State Sequence using the Viterbi Algorithm
print("c) Finding Most Likely State Sequence (Viterbi Algorithm)")
# The model.predict() method uses the Viterbi algorithm.
predicted_states_int = model.predict(obs_seq_int)

# Map the integer state indices back to their string names
states_map_inv = {i: state for i, state in enumerate(states)}
predicted_states_str = [states_map_inv[s] for s in predicted_states_int]

print(f"For Observation Sequence: {obs_seq_str}")
print(f"The Most Likely State Sequence is: {predicted_states_str}\n")


# d) Generate sequences and estimate the empirical probability
print("d) Estimating Empirical Probability and Comparing")
n_sequences_to_generate = 10000
sequence_length = len(obs_seq_int)

# The model.sample() method returns (observations, states)
generated_sequences, _ = model.sample(n_samples=sequence_length, n_sequences=n_sequences_to_generate)

# Count how many of the generated sequences match our target sequence
match_count = 0
target_array = obs_seq_int.flatten()


# We need to iterate through the sequences correctly.
# The shape will be (n_sequences, sequence_length).
for i in range(n_sequences_to_generate):
    # Extract the i-th sequence
    current_sequence = generated_sequences[i*sequence_length:(i+1)*sequence_length].flatten()
    if np.array_equal(current_sequence, target_array):
        match_count += 1

# Calculate the empirical probability
empirical_probability = match_count / n_sequences_to_generate

print(f"Generated {n_sequences_to_generate} sequences of length {sequence_length}.")
print(f"Found {match_count} matches for the sequence {obs_seq_str}.")
print(f"The empirical probability is: {empirical_probability:.6f}\n")


# Compare
print(f"Theoretical Probability (from Forward Algorithm): {prob_theoretical:.6f}")
print(f"Empirical Probability (from {n_sequences_to_generate} samples):   {empirical_probability:.6f}")
