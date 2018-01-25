
from GeneralRelativity.CreateGrTensors import *
from CurvedScalarWave.CreateCswTensors import *

def compute_csw_dt_psi(dim: int) -> float:
    dt_psi = 0.
    dt_psi += (1+make_gamma1()) * np.dot(make_shift(dim), make_d_psi(dim))
    dt_psi -= make_lapse() * make_pi()
    dt_psi -= make_gamma1() * np.dot(make_shift(dim), make_phi(dim))
    return dt_psi

def compute_csw_dt_pi(dim: int) -> float:
    dt_pi = 0.
    dt_pi += make_lapse() * make_trace_k() * make_pi()
    print(dt_pi)
    dt_pi += np.dot(make_shift(dim), make_d_pi(dim))
    print(dt_pi)
    dt_pi += make_lapse() * np.dot(make_trace_spatial_christ_2(dim), make_phi(dim))
    print(dt_pi)
    dt_pi += make_gamma2() * make_gamma1() * np.dot(make_shift(dim), make_d_psi(dim) - make_phi(dim))
    print(dt_pi)
    dt_pi -= make_lapse() * np.einsum("kj,kj", make_inv_g(dim), make_d_phi(dim))
    print(dt_pi)
    dt_pi -= np.einsum("kj,k,j", make_inv_g(dim), make_d_lapse(dim), make_phi(dim))
    print(dt_pi)
    return dt_pi

def compute_csw_dt_phi(dim: int) -> np.ndarray:
    dt_phi = np.zeros(dim)
    dt_phi += np.einsum("k,ki->i", make_shift(dim), make_d_phi(dim))
    dt_phi -= make_lapse() * make_d_pi(dim)
    dt_phi += make_gamma2() * make_lapse() * make_d_psi(dim)
    dt_phi -= make_pi() * make_d_lapse(dim)
    dt_phi += np.einsum("j, ij -> i", make_phi(dim), make_d_shift(dim))
    dt_phi -= make_gamma2() * make_lapse() * make_phi(dim)
    return dt_phi


