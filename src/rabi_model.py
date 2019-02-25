import datetime
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import qutip
#QuTiP control modules
import qutip.control.pulseoptim as cpo
import scipy


from src.utils import ground_state


def QRM_free_term(N, Omega, omega_0):
    """Free term of the QRM Hamiltonian.
    
    Parameters
    ----------
    N : int
        The dimension of the truncated Fock space
    Returns
    -------
    A qutip.qObj matrix-like object.
    """
    H0 = Omega / 2 * qutip.tensor(qutip.qeye(N), qutip.sigmaz())
    H0 += omega_0 * qutip.tensor(qutip.create(N) * qutip.destroy(N), qutip.qeye(2))
    return H0


def QRM_interaction_term(N):
    return -qutip.tensor(qutip.create(N) + qutip.destroy(N), qutip.sigmax())


def QRM_full(N, Omega, omega_0, lambda_):
    """Full QRM Hamiltonian."""
    H = QRM_free_term(N, Omega, omega_0)
    H += lambda_ * QRM_interaction_term(N)
    return H

