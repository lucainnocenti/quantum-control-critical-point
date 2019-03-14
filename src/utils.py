import datetime
import logging
import importlib
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar
import qutip


def ground_state(H):
    """Shortcut to compute the ground state of an Hamiltonian.
    
    Parameters
    ----------
    H : qutip.Qobj matrix
    """
    return H.eigenstates(eigvals=1)[1][0]

def plot(fun, times, *args, **kwargs):
    plt.figure()
    plt.plot(times, [fun(t) for t in times], *args, **kwargs)


def autonumber_filename(filename):
    base, ext = os.path.splitext(filename)
    if not os.path.isfile(filename):
        return filename
    idx = 0
    while os.path.isfile('{}({:02}){}'.format(base, idx, ext)):
        idx += 1
    filename = base + '({:02})'.format(idx) + ext
    return filename


def timestamp():
    return datetime.datetime.now().strftime('%y%m%dh%Hm%Ms%S')


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple, np.ndarray)):
            for j in flatten(i):
                yield j
        else:
            yield i


def basic_logger_configuration(filename=None, toconsole=False, reset=False):
    if filename is None and not toconsole:
        raise ValueError('At least one of tofile and toconsole must be true.')
    if reset:
        logging.shutdown()
        importlib.reload(logging)
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s]"
                                     "[%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    if filename is not None and isinstance(filename, str):
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        rootLogger.addHandler(fileHandler)
    if toconsole:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
