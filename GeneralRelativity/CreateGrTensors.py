""" Helper functions which create GR tensors with 'random' values. The values
used here should match identically with those used in SpECTRE"""

import numpy as np

def make_lapse() -> float:
    return 3.

def make_shift(dim) -> np.ndarray:
    return np.array(range(1,dim+1))

def make_g(dim) -> np.ndarray:
    x = np.array(range(1,dim+1))
    return np.tensordot(x,x,0)

def make_inv_g(dim) -> np.ndarray:
    x = np.array(range(7,dim+7))
    return np.tensordot(x,x,0)

def make_dt_lapse() -> float:
    return 5.

def make_dt_shift(dim) -> np.ndarray:
    return dim*np.arange(1,dim+1)

def make_dt_g(dim) -> np.ndarray:
    return np.array([[i + j for j in range(dim)] for i in range(dim)])

def make_d_lapse(dim) -> np.ndarray:
    return 2.5 * np.arange(1,dim+1)

def make_d_shift(dim) -> np.ndarray:
    return np.array( [[ 3 * ( j + 1.) - i + 4 for j in range(dim)] for i in range(dim)]  )

def make_d_g(dim) -> np.ndarray:
    return np.array( [[[ 3 * (i+1)*(j+1) + k for j in range(dim)] for i in range(dim)] for k in range(dim)])

def make_psi(dim) -> np.ndarray:
    return np.array( [[ -2 * (i+2)*(j+2) for j in range(dim+1)] for i in range(dim+1)])

def make_spacetime_christoffel_2(dim) -> np.ndarray:
    return np.array( [[[ (i+2) * (j+1) * (k+2) for j in range(dim)] for i in range(dim)] for k in range(dim)])

def make_spatial_christoffel_2(dim) -> np.ndarray:
    return np.array( [[[ 4 * (i+1) * (j+1) - k for j in range(dim)] for i in range(dim)] for k in range(dim)])

def make_inv_psi(dim) -> np.ndarray:
    return np.array( [[ -2 * (i-2)*(j+2)+1 for j in range(dim)] for i in range(dim)])

def make_d4_lapse(dim) -> np.ndarray:
    l = np.zeros(dim+1)
    l[0] = make_dt_lapse()
    l[1:] = make_d_lapse(dim);
    return l

def make_d4_shift(dim) -> np.ndarray:
    l = np.zeros([dim+1, dim])
    l[0] = make_dt_shift(dim)
    l[1:] = make_d_shift(dim)
    return l

def make_d4_g(dim) -> np.ndarray:
    l = np.zeros([dim+1, dim, dim])
    l[0] = make_dt_g(dim)
    l[1:] = make_d_g(dim)
    return l

def d4psi(dim) -> np.ndarray:
    return np.array([[[(i+1)*(j+1)*(k+3) for j in range(dim)] for i in range(dim)] for k in range(dim)] )

def d3psi(dim) -> np.ndarray:
    return np.array([[[(i+1)*(j+1)*(k+3) for j in range(dim+1)] for i in range(dim+1)] for k in range(dim)] )

def make_trace_k() -> float:
    return 5.

def make_trace_spatial_christ(dim) -> np.ndarray:
    return np.array([3*i - 2 for i in range(dim)])

def make_dt_psi(dim) -> np.ndarray:
    return np.array([[i*j + 0.5 for j in range(dim+1)] for i in range(dim+1)])
