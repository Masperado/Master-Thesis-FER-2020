import numpy as np
from scipy.linalg import logm, expm, inv

# Tests out distance on manifold of SPD matrices on locally tangent space

# Generate SPD matrices
m1 = np.mat(np.random.rand(6,7))
M1 = m1*m1.T
m2 = np.mat(np.random.rand(6,7))
M2 = m2*m2.T

# Calculate error on tangent space of M1
cM1 = np.mat( expm(0.5 * logm(M1)) )
cM1_i = np.mat( expm(0.5 * logm(inv(M1))) )
L12 = cM1 * logm(cM1_i * M2 * cM1_i) * cM1

# Print error
#print(L12)
#print("----------------------")

# Add error to M1 to try and recover M2
M2_r = cM1 * expm(cM1_i * L12 * cM1_i) * cM1

# New error between M2 and M2_r should be close to zero
cM1 = np.mat( expm(0.5 * logm(M2)) )
cM1_i = np.mat( expm(0.5 * logm(inv(M2))) )
L12r = cM1 * logm(cM1_i * M2_r * cM1_i) * cM1

# Should be close to zero
#print(L12r)

if not np.any(np.absolute(L12r) > 10e-11):
    print("Passed")


# PASSED