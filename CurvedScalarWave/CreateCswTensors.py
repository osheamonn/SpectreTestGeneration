
import numpy as np

def make_gamma1() -> float:
    return 5.8

def make_gamma2() -> float:
    return -4.3

def make_d_psi(dim : int) -> np.ndarray:
    return np.array([(2*i + 0.5) for i in range(dim)])

def make_phi(dim : int) -> np.ndarray:
    return np.array([(-3*i + 0.5) for i in range(dim)])

def make_psi() -> float:
    return 9.8

def make_pi() -> float:
    return -1.6

def make_d_phi(dim : int) -> np.ndarray:
    return np.array(
        [[5. * (i - 0.5) + 0.5 + j for j in range(dim)] for i in range(dim)]
    )

def make_d_pi(dim : int) -> float:
    return np.array([2.5*i + 0.5 for i in range(dim)])
