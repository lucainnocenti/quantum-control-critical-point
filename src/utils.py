import datetime
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import progressbar
import qutip


def ground_state(H):
    """Shortcut to compute the ground state of an Hamiltonian.
    
    Parameters
    ----------
    H : qutip.Qobj matrix
    """
    return H.eigenstates(eigvals=1)[1][0]
