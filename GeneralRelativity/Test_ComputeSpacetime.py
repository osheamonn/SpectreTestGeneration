
from GeneralRelativity.CreateGrTensors import *

def compute_spacetime_metric(dim) -> np.ndarray:
    assert 0 < dim and dim < 4
    psi = np.zeros([dim+1, dim+1])
    psi[0,0] = -make_lapse()**2 + np.einsum('i,ij,j',make_shift(dim), make_g(dim), make_shift(dim))
    psi[1:,1:]=make_g(dim)
    psi[1:,0]=np.einsum('ij,j',make_g(dim), make_shift(dim))
    psi[0,1:]=psi[1:,0] #Ensure symmetry
    return psi

def compute_inverse_spacetime_metric(dim) -> np.ndarray:
    assert(0 < dim and dim < 4)
    psi = np.zeros([dim+1, dim+1])
    psi[0,0] = - 1/make_lapse()**2
    psi[1:,0] = make_shift(dim)/make_lapse()**2
    psi[0,1:] = psi[1:,0]
    psi[1:,1:] = make_g(dim) - np.tensordot(make_shift(dim), make_shift(dim), 0) / make_lapse()**2
    return psi

def compute_spacetime_derivatives_of_spacetime_metric(dim) -> np.ndarray:
    assert(0 < dim and dim < 4)
    g = np.zeros([dim+1, dim+1, dim+1])
    g[:,0,0] = -2 * make_lapse() * make_d4_lapse(dim) + 2 * np.einsum("mn,m,an->a",make_g(dim), make_shift(dim), make_d4_shift(dim)) + np.einsum("m,n,amn->a",make_shift(dim), make_shift(dim), make_d4_g(dim))
    g[:,1:,0] = np.einsum("mi,am->ai", make_g(dim), make_d4_shift(dim)) + np.einsum("m,ami->ai", make_shift(dim), make_d4_g(dim))
    g[:,0,1:]=g[:,1:,0]
    g[:,1:,1:]=make_d4_g(dim)
    return g

def compute_christoffel(d_metric) -> np.ndarray:
    dim = d_metric.shape[0]
    print(dim)
    return 0.5 * np.array([[[d_metric[b,c,a] + d_metric[a,c,b] - d_metric[c,a,b] for b in range(dim)] for a in range(dim)] for c in range(dim)])

def compute_pi(dim) -> np.ndarray:
    assert(0 < dim and dim < 4)
    d_psi = compute_spacetime_derivatives_of_spacetime_metric(dim)
    pi = d_psi[0,:,:]
    pi -= np.tensordot(make_shift(dim), d3psi(dim), [0,0])
    pi /= -make_lapse()
    return pi

def compute_gauge_constraint(dim) -> np.ndarray:
    assert(0 < dim and dim < 4)
    H = np.zeros(dim+1)
    shift_dot_d_shift = np.einsum("k,ki->i", make_shift(dim), make_d_shift(dim))
    H[1:] = np.einsum("ij,j->i",make_g(dim), make_dt_shift(dim) - shift_dot_d_shift) / make_lapse()**2 + 1./make_lapse() * make_d_lapse(dim) - make_trace_spatial_christ(dim)
    H[0] = -make_dt_lapse() / make_lapse() + 1/make_lapse() * np.tensordot(make_shift(dim), make_d_lapse(dim), [0,0]) + np.tensordot(make_shift(dim), H[1:],[0,0]) - make_lapse() * make_trace_k()
    return H

def compute_ext_curvature(dim) -> np.ndarray:
    assert(0 < dim and dim < 4)
    ext_curve = np.zeros([dim,dim])
    ext_curve = np.tensordot(make_shift(dim), make_d_g(dim), [0,0]) + \
                np.einsum("ki,jk->ij", make_g(dim), make_d_shift(dim)) + \
                np.einsum("kj,ik->ij", make_g(dim), make_d_shift(dim)) -\
                make_dt_g(dim)
    ext_curve *= 0.5 / make_lapse()
    return ext_curve
