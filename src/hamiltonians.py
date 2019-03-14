import datetime
import logging
import os
import pickle
import functools
import sys
import numbers
import time

import numpy as np
import pandas as pd
import qutip

from src.utils import ground_state
import src.protocol_ansatz as protocol_ansatz


class TimeDependentHamiltonian:
    def __init__(self, model_parameters):
        """Should involve initialization of core model parameters.
        
        Define time-independent and time-dependent components (`H0` and `H1`),
        name of the model, and time-dependent protocol (as ProtocolAnsatz obj
        instance).
        """
        self.H0 = None
        self.H1 = None
        self.td_protocol = None
        self.model_parameters = None
        raise NotImplementedError('Children must override this.')
    
    def hamiltonian(self, parameter):
        """Return the Hamiltonian for a given value of the parameter."""
        return self.H0 + parameter * self.H1
    
    def ground_state(self, parameter):
        return ground_state(self.hamiltonian(parameter))
    
    def _parse_td_protocol(self, td_protocol_name):
        """Input must be a string, parsed to figure out the protocol to use."""
        if td_protocol_name == 'doublebang':
            self.td_protocol = protocol_ansatz.DoubleBangProtocolAnsatz()
        elif td_protocol_name == 'bangramp':
            self.td_protocol = protocol_ansatz.BangRampProtocolAnsatz()
        elif td_protocol_name == 'triplebang':
            self.td_protocol = protocol_ansatz.TripleBangProtocolAnsatz()
        else:
            raise ValueError('Other protocols must be given explicitly.')
    
    def critical_hamiltonian(self):
        return self.hamiltonian(self.critical_value)
    
    def critical_ground_state(self):
        return ground_state(self.critical_hamiltonian)
    
    def evolve_state(self, state, tlist, td_protocol_parameters,
                     return_all_states=False):
        if self.td_protocol is None:
            raise ValueError('The protocol must be specified first.')
        if isinstance(tlist, numbers.Number):
            tlist = np.linspace(0, tlist, 40)

        protocol = self.td_protocol
        td_fun = protocol.time_dependent_fun(td_protocol_parameters)
        # the following loop was added because often the ODE solver used by
        # qutip.mesolve got stuck. In those cases increasing the number of
        # steps seemed to always fix the problem, so this is a quick and dirty
        # solution to work around the problem.
        trial_idx = 0
        results = None
        while (trial_idx < 3 and results is None):
            try:
                results = qutip.mesolve(
                    H=[self.H0, [self.H1, td_fun]],
                    rho0=state,
                    tlist=tlist
                ).states
            except Exception:
                logging.info('The ODE solver seems to have got stuck. Trying '
                             'to double the number of time steps.')
                trial_idx += 1
                tlist = np.linspace(tlist[0], tlist[-1], 2 * len(tlist))

        if trial_idx == 3:
            logging.info('THE SOLVER FUCKING SUCKS')
            return state

        
        if return_all_states:
            return results
        else:
            return results[-1]
