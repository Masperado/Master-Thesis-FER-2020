import numpy as np
from scipy.linalg import expm, logm, inv
from liegroups.numpy import SE3, SO3

POS = range(0,3)
ROT = range(3,6)

# TENSOR RELATED FUNCTIONS
# --------------------------------------------------------------------------------------------------------------

def unfold(T, mode):
    "http://tensorly.org/dev/_modules/tensorly/base.html#unfold"
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))


def fold(u_T, mode, shape):
    "http://tensorly.org/dev/_modules/tensorly/base.html#fold"
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(u_T, full_shape), 0, mode)


def mode_dot(T, M, mode):
    fold_mode = mode
    new_shape = list(T.shape)

    new_shape[mode] = M.shape[0]

    res = np.array(np.dot(M, unfold(T, mode)))
    return fold(res, fold_mode, new_shape)

def symmat2vec(M):
    # Vectorization of a symmetric matrix

    N = np.size(M,0)
    v = np.diag(M)
    for n in range(1,N):
        h = list(np.sqrt(2)*np.diag(M,n))
        v = np.append(v,h) # Mandel notation

    print(v)
    return v


def SPD_error(M1, M2):
    """
    Distance between two symmetric positive semidefinite matrices on a Riemannian manifold
    """
    cM1 = np.mat(expm(0.5 * logm(M1)))
    cM1_i = np.mat(expm(0.5 * logm(inv(M1))))
    return cM1 * logm(cM1_i * M2 * cM1_i) * cM1

def pose_error(p1, q1, p2, q2):
    """
    Twist representing the distance between two poses in the world frame
    The transform between poses is returned in the pose 1 frame, then we rotate it back to world frame
    """
    r1 = SO3.from_quaternion(q1, 'xyzw')
    r2 = SO3.from_quaternion(q2, 'xyzw')

    t1 = SE3(r1, p1)
    t2 = SE3(r2, p2)

    xi = SE3.log((t1.inv().dot(t2)))

    return r1.dot(xi[POS]),r1.dot(xi[ROT])