import datetime
import logging
import os
import pickle
import sys
import time

import numpy as np
import scipy
import qutip

from src.utils import ground_state


def J_plus_operator(num_spins, total_j, spin_value=0.5):
    dim_space = int(2 * total_j) + 1
    Jp_matrix = np.zeros(shape=[dim_space] * 2)
    for m in range(int(2 * total_j)):
        m_orig = m - total_j
        Jp_matrix[m + 1, m] = np.sqrt(total_j * (total_j + 1) -
                                      m_orig * (m_orig + 1))
    return qutip.Qobj(Jp_matrix)


def J_minus_operator(num_spins, total_j, spin_value=0.5):
    dim_space = int(2 * total_j) + 1
    Jm_matrix = np.zeros(shape=[dim_space] * 2)
    for m in range(1, int(2 * total_j) + 1):
        m_orig = m - total_j
        Jm_matrix[m - 1, m] = np.sqrt(total_j * (total_j + 1) -
                                      m_orig * (m_orig - 1))
    return qutip.Qobj(Jm_matrix)


def Jx_operator(num_spins, total_j, spin_value=0.5):
    return (J_plus_operator(num_spins, total_j, spin_value) +
            J_minus_operator(num_spins, total_j, spin_value))


def Jz_operator(num_spins, total_j, spin_value=0.5):
    return qutip.Qobj(np.diag(np.arange(-total_j, total_j + 1)))


def LMG_free_term(num_spins, total_j=None):
    if total_j is None:
        total_j = num_spins / 2
    return Jz_operator(num_spins, total_j)


def LMG_interaction_term(num_spins, g_value=1., total_j=None):
    """Interaction term of the LMG Hamiltonian.

    Given in the standard form - g / 4N * Sx^2
    """
    if total_j is None:
        total_j = num_spins / 2
    Jx_squared = Jx_operator(num_spins, total_j) ** 2
    return - g_value / (4 * num_spins) * Jx_squared


def LMG_full_hamiltonian(num_spins, g_value=1., total_j=None):
    return (LMG_free_term(num_spins, total_j) +
            LMG_interaction_term(num_spins, g_value, total_j))


def prepare_hamiltonian_and_states_for_optimization(num_spins, total_j=None):
    H0 = LMG_free_term(num_spins, total_j=total_j)
    H1 = LMG_interaction_term(num_spins, g_value=1., total_j=total_j)
    initial_state = ground_state(H0)
    target_state = ground_state(H0 + H1)
    return dict(H0=H0, H1=H1,
                initial_state=initial_state, target_state=target_state)
