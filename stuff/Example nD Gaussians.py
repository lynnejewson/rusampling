import numpy as np
from ru import Ru
import matplotlib.pyplot as plt

def logf(x):
    return -0.5 * np.sum(x**2, axis=1)

def pa_exact(r, d):
    acceptance_area = (2*np.pi)**(d/2) / (1 + r*d)
    total_area = (4*(1 + r*d)/(np.e*r))**(d/2)
    return acceptance_area / total_area

def rectangle_exact(r, d):
    # f is the same in all dimensions, so this is the same on all sides of the rectangle
    b = ((1 + r*d)/(np.e * r))**0.5
    return b


n = 100000
d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
probabilities = []
probabilities_exact = []
r = 0.5

for d_i in d:
    probabilities_exact.append(pa_exact(r, d_i))
    t = Ru(logf, d=d_i, r=r)
    out = t.rvs_detail(n=n)
    probabilities.append(out['pa'])

    # output information to check how accurate the optimisation is
    rectangle_bound_estimate = t.v_max[0]
    rectangle_bound_exact = rectangle_exact(r, d_i)
    print(f'Rectangle estimated at {rectangle_bound_estimate}. Actually at {rectangle_bound_exact}.')

# plot results
plt.plot(d, probabilities, color='black', label='In trials')
plt.plot(d, probabilities_exact, color='red', label='Exact')
plt.title('Acceptance Probability')
plt.xlabel('d')
plt.legend()
plt.show()