import numpy as np
import random
import matplotlib.pyplot as plt

#2.1
X1 = np.random.poisson(1, 1000)
X2 = np.random.poisson(2, 1000)
X3 = np.random.poisson(5, 1000)
X4 = np.random.poisson(10, 1000)

#2.2 
lambdas = [1, 2, 5, 10]
X_rand = np.array([np.random.poisson(random.choice(lambdas)) for _ in range(1000)])

#a
plt.figure(figsize=(12, 10))

datasets = [(X1, 'Poisson(1)'), (X2, 'Poisson(2)'), (X3, 'Poisson(5)'), (X4, 'Poisson(10)'), (X_rand, 'Randomized Poisson')]
for i, (data, title) in enumerate(datasets, 1):
    plt.subplot(3, 2, i)
    plt.hist(data, bins=range(0, max(data)+2), color='blue', alpha=0.8, edgecolor='black')
    plt.title(title)
    plt.xlabel('Număr apeluri pe oră')
    plt.ylabel('Frecvență')
    plt.tight_layout()

plt.show()

#Cu cat valoarea λ este mai mare, distributia Poisson este pe un esantion mai mare,
#si varful acesteia corespunde valorii lambda.

#Asta ne spune ca variabilitatea valorii lamba, poate face ca rezultatele sa fie neasteptate si
#imprevizibile



