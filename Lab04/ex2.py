import numpy as np
import random
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

size = 5
np.random.seed(0)
original = np.random.randint(0, 2, (size, size))

noisy = original.copy()
num_noisy = int(size * size * 0.1)  # 10% noisy pixels
indices = random.sample(range(size * size), num_noisy)
for idx in indices:
    i, j = divmod(idx, size)
    noisy[i, j] = 1 - noisy[i, j]  # flip bit

print("Original Image:\n", original)
print("Noisy Image:\n", noisy)

model = MarkovNetwork()

nodes = [(i, j) for i in range(size) for j in range(size)]
model.add_nodes_from(nodes)

edges = []
for i in range(size):
    for j in range(size):
        if i + 1 < size: edges.append(((i, j), (i + 1, j)))
        if j + 1 < size: edges.append(((i, j), (i, j + 1)))

model.add_edges_from(edges)

lambda_reg = 2.0

factors = []
for i in range(size):
    for j in range(size):
        noisy_val = noisy[i, j]
        potential = np.array([np.exp(-lambda_reg * (0 - noisy_val) ** 2),
                              np.exp(-lambda_reg * (1 - noisy_val) ** 2)])
        factor = DiscreteFactor(variables=[(i, j)],
                                cardinality=[2],
                                values=potential)
        factors.append(factor)

same = np.exp(-0)
diff = np.exp(-1)
pairwise_potential = np.array([[same, diff],
                               [diff, same]])

for edge in edges:
    factor = DiscreteFactor(variables=[edge[0], edge[1]],
                            cardinality=[2, 2],
                            values=pairwise_potential)
    factors.append(factor)

model.add_factors(*factors)

bp = BeliefPropagation(model)
map_estimate = bp.map_query(variables=nodes)

denoised = np.zeros((size, size), dtype=int)
for (i, j), val in map_estimate.items():
    denoised[i, j] = val

print("Denoised Image:\n", denoised)
