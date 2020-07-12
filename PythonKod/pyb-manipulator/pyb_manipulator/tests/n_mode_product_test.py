import numpy as np
from utils import math_utils as mu
import tensorly

# Compare n-mode product code  to tensorly implementation
T = np.random.rand(5,5,5)
M = np.random.rand(5,5)

res_1 = mu.mode_dot(T, M, 0)

res_2 = tensorly.tenalg.mode_dot(T,M,0)

# print out difference
print(res_2 - res_1)

# PASSED