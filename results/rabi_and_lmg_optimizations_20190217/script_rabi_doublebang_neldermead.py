import datetime
import os
import pickle
import functools
import sys
import time
import logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                 "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

import numpy as np
import pandas as pd
import progressbar
import qutip
import qutip.control.pulseoptim as cpo
import scipy

if '../../' not in sys.path:
    sys.path.append('../../')
import src.rabi_model as rabi_model
import src.optimization as optimization
from src.utils import ground_state


# model parameters
N = 100
Omega = 100
omega_0 = 1
lambda_c = np.sqrt(Omega * omega_0) / 2.
# build Hamiltonians
H0 = rabi_model.QRM_free_term(N, Omega, omega_0)
H1 = rabi_model.QRM_interaction_term(N)
# compute initial and target states
initial_state = ground_state(H0)
target_state = ground_state(H0 + lambda_c * H1)
# run optimization
times_to_try = np.linspace(0.1, 4, 100)
results = np.zeros(shape=[len(times_to_try), 5])

for idx, tf in enumerate(times_to_try):
    parametrized_model = functools.partial(
        optimization.make_doublebang_pulse_fun, tf=tf)
    logging.info('Starting optimization for tf={}'.format(tf))
    result = optimization.optimize_model_parameters(
        hamiltonians=[H0, H1],
        initial_state=initial_state, target_state=target_state,
        evolution_time=tf,
        parametrized_model=parametrized_model,
        initial_parameters=[1, tf / 2, 1],
        optimization_method='Nelder-Mead', stfu=True,
        parameters_constraints=[[-50, 50], [0, tf], [-50, 50]]
    )
    obtained_fidelity = (1 - result.fun)**2
    results[idx] = [tf, obtained_fidelity, *result.x]
    logging.info('    Result: {}'.format(obtained_fidelity))
results = pd.DataFrame(results, columns=['tf', 'fid', 'y0', 't1', 'y1'])

results.to_csv('rabi_Omega100_doublebang_neldermead.csv')
