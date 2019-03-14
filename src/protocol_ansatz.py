import datetime
import logging
import os
import pickle
import functools
import sys
import numbers
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_segment(x0, x1, y0, y1, t):
    """Return the linear function interpolating the given points."""
    return y0 + (t - x0) / (x1 - x0) * (y1 - y0)


def _make_CRAB_pulse_correction_fun(Ak, Bk, nuk, tf, normalising_pulse=None):
    """Return function that gives CRAB correction at any given time.

    Outputs a function that for any given time returns the CRAB correction.
    It is by design vanishing at t=0 and t=tf, and intended to be multiplied
    by the linear baseline before the optimization.

    Parameters
    ----------
    nuk : numpy array of floats
        Should be a random float between -0.5 and 0.5, that will be used to
        perturb the corresponding Fourier component of the pulse.
    """
    Ak = np.asarray(Ak)
    Bk = np.asarray(Bk)
    nuk = np.asarray(nuk)
    # there must be the same number of elements in Ak, Bk, nuk
    if len(Bk) != len(Ak) or len(Ak) != len(nuk):
        raise ValueError('There must be the same number of Ak, Bk, nuk.')
    N = len(Ak)

    # random_frequencies = 2 * np.pi * np.arange(1, N + 1) * nuk / tf
    random_frequencies = 2 * np.pi * np.arange(1, N + 1) * (1 + nuk) / tf
    # logging.debug('Actual CRAB frequencies: {}'.format(random_frequencies))

    # make normalizer function, which enforces the corrections to be vanishing
    # at the initial and final times. This might heavily affect results.
    if normalising_pulse is None:
        def normalising_pulse(t):
            return 0.1 * t * (t - tf)
    # build the truncated, normalised, randomised fourier series

    def pulse(t):
        return 1 + normalising_pulse(t) * np.sum(
            Ak * np.sin(random_frequencies * t) +
            Bk * np.cos(random_frequencies * t)
        )
    return pulse


def _make_CRAB_ramp_fun(Ak, Bk, nuk, tf, y0, y1, normalising_pulse=None):
    """Return the time-dependent component of the CRAB Hamiltonian as a function.
    
    Parameters
    ----------
    Ak, Bk : arrays of floats
        Parameteres of truncated fourier series.
    nuk : array of floats
        The (usually random) frequency of the CRAB Ansatz
    tf : float
        Total evolution time
    y0, y1 : floats
        Initial and final value of the pulse.
    """
    CRAB_correction = _make_CRAB_pulse_correction_fun(
        Ak, Bk, nuk, tf, normalising_pulse=normalising_pulse)
    def final_ramp_fun(t, *args):
        return CRAB_correction(t) * linear_segment(x0=0, x1=tf, y0=0, y1=y1, t=t)
    return final_ramp_fun


def make_CRAB_final_ramp_fun(parameters, nuk, tf, y0, y1,
                             normalising_pulse=None):
    """Return overall pulse function used for CRAB model.

    Parameters
    ----------
    parameters : 1D np.array
        Contains parameters A and B in the standard CRAB ansatz. It is
        assumed to have length 2 * n where n is the number of frequencies.
    """
    parameters = np.asarray(parameters)
    Ak, Bk = parameters.reshape((2, len(parameters) // 2))
    return _make_CRAB_ramp_fun(Ak, Bk, nuk, tf, y0, y1,
                               normalising_pulse=normalising_pulse)


def make_linear_interpolation_fun(pairs):
    """Build and return function interpolating the provided input points.

    Parameters
    ----------
    pairs : list of tuples
        The returned function interpolates linearly between the values.
        Every element of this list should be a pair of numeric values, the
        first one representing the x-axis and the second one the y-axis.
    
    Returns
    -------
    A function returning a single numeric output for each numeric input.
    The function returns a fixed value of 200 if given an input outside of its
    range of definition.
    The returned function also accepts unnamed optional arguments that are not
    actually used. This is to accomodate the requirements of qutip.mesolve.

    Example
    -------
    >>> make_linear_interpolation_fun([(0, 0), (2, 2), (4, 2)])(
    1.0
    """
    OUT_OF_BOUNDARIES_VALUE = 200
    def fun(t, *args):
        for idx, (x, y) in enumerate(pairs):
            if t >= x:
                # this should not happen: we are outside the region in which
                # the function is defined
                if idx == len(pairs) - 1:  # last element of the list of pair
                    # we make sure that are significantly outside of the
                    # interval of definition, to avoid numerical shenaningans
                    # around the final border
                    if t > 1.1 * x:
                        return OUT_OF_BOUNDARIES_VALUE
                    else:
                        return y
                if t >= pairs[idx + 1][0]:
                    continue
                # else we are in the correct bracket
                slope = (pairs[idx + 1][1] - y) / (pairs[idx + 1][0] - x)
                return y + (t - x) * slope
        # if we are still here, then the input is smaller than all the x values
        # with which the function has been defined. In this case just return 
        # the fixed out of boundaries value
        return OUT_OF_BOUNDARIES_VALUE
    return fun


def make_bangramp_pulse_fun(parameters, tf):
    """Return bangramp pulse function.

    The first parameter is a list containing the parameters that will be
    updated during the optimization. All the other parameters are to be
    considered "metaparameters", and are fixed before the optimization begins.
    
    Pulse shape:
        For 0 <= t <= t1, constant function with height y0.
        For t >= t1 (and presumably t <= tf), linear ramp going from y1 to y2.
    """
    y0, t1, y1, y2 = parameters
    def fun(t, *args):
        if 0 <= t <= t1:
            return y0
        elif t1 < t <= tf:
            return y1 + (t - t1) / (tf - t1) * (y2 - y1)
        else:
            return 200.
    return fun


def make_doublebang_pulse_fun(parameters, tf):
    """Return double-bang pulse function.

    For 0 <= t <= t1, constant height y0.
    For t >= t1, constant height y1.
    """
    y0, t1, y1 = parameters
    def fun(t, *args):
        if 0 <= t <= t1:
            return y0
        elif t1 < t <= tf:
            return y1
        else:
            return 200.
    return fun


class ProtocolAnsatz:
    """Prototype for protocol pulse shapes (e.g. doublebang, bangramp, CRAB).

    Attributes
    ----------
    
    """
    
    def __init__(self, name, pars_names, hyperpars_names=None):
        """
        Parameters
        ----------
        name : string
            Name of the protocol
        pars_names: list of strings
            Each parameter should be associated with a name, used to name
            the columns in the final optimization results. This parameter is
            also used to determine the number of parameters accepted by the
            protocols
        hyperpars_names: list of strings
            Same as pars_names, but for the hyperparameters.
        """
        self.name = name
        self.pars_names = pars_names
        self.hyperpars = OrderedDict()
        for hyperpar_name in hyperpars_names:
            self.hyperpars[hyperpar_name] = None
        self.pars_constraints = [[None, None]] * len(self.pars_names)

        self.total_height_constraint = None
        self.out_of_boundaries = False

        logging.debug('Initializing ProtocolAnsatz instance "{}"'.format(self.name))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        raise ValueError('Invalid comparison')
    
    def __str__(self):
        return self.name

    def fill_hyperpar_value(self, **kwargs):
        """Fill in hyperparameters.
        
        Each input key should match one of the keys of self.hyperpars
        """
        for name, value in kwargs.items():
            try:
                self.hyperpars[name]
            except KeyError:
                raise ValueError('{} is not a recognised hyperparameter name.'.format(name))
            # otherwise just use the value
            self.hyperpars[name] = value
            logging.debug('Setting hyperparameter {}={}'.format(name, value))

    def protocol_shape(self, parameters):
        raise NotImplementedError('Subclasses must override protocol_shape!')

    def time_dependent_fun(self, parameters):
        """Provided all hyperparameters have been filled, generate pulse shape.

        The length of the input must match the length of self.pars_names, and
        all hyperparameters must have been filled prior to calling this.
        """
        # verify that all hyperparameters have been given
        for name, hyperpar in self.hyperpars.items():
            if hyperpar is None:
                raise ValueError('The hyperparameter {} has not been given yet.'.format(name))
        fun = self.protocol_shape(parameters)
        # replace function to checks total height constraints, if needed
        if self.total_height_constraint is not None:
            fun = self.add_total_height_constraint_check(fun)
        # return the function (which for each time gives a number)
        return fun
    
    def add_parameter_constraints(self, **kwargs):
        for par_name, constraint in kwargs.items():
            if par_name not in self.pars_names:
                raise ValueError('"{}" is not a recognised parameter name'.format(par_name))
            self.pars_constraints[self.pars_names.index(par_name)] = constraint

    def are_pars_in_boundaries(self, pars):
        """Check whether the parameters are in the range they should be in.

        Returns True if the given parameters are within the constraints.
        """
        for par, constraint in zip(pars, self.pars_constraints):
            if constraint[0] is not None and par < constraint[0]:
                return False
            elif constraint[1] is not None and par > constraint[1]:
                return False
        return True
    
    def set_total_height_constraints(self, boundaries):
        self.total_height_constraint = boundaries
    
    def add_total_height_constraint_check(self, fun):
        def flagged_fun(time, *args):
            value = fun(time)
            if (value < self.total_height_constraint[0] or
                    value > self.total_height_constraint[1]):
                self.out_of_boundaries = True
            return value
        return flagged_fun
    
    def plot(self, pars, ax=None, *args, **kwargs):
        fun = self.time_dependent_fun(pars)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        times = np.linspace(0, self.hyperpars['tf'])
        heights = [fun(t) for t in times]
        ax.plot(times, heights, *args, **kwargs)


class DoubleBangProtocolAnsatz(ProtocolAnsatz):
    def __init__(self):
        super().__init__(
            name='doublebang',
            pars_names=['y0', 't1', 'y1'],
            hyperpars_names=['tf']
        )
        self.num_parameters_to_save = 4
    
    def protocol_shape(self, pars):
        return make_doublebang_pulse_fun(pars, self.hyperpars['tf'])
    
    def fill_hyperpar_value(self, **kwargs):
        """Automatically add constraints whenever tf is filled."""
        for par_name, par_value in kwargs.items():
            if par_name == 'tf':
                self.add_parameter_constraints(t1=[0, par_value])
        super().fill_hyperpar_value(**kwargs)
    
    def constrain_intensities(self, boundaries):
        """Add constraints to all non-time parameters."""
        self.add_parameter_constraints(y0=boundaries, y1=boundaries)


class BangRampProtocolAnsatz(ProtocolAnsatz):
    def __init__(self):
        super().__init__(
            name='bangramp',
            pars_names=['y0', 't1', 'y1', 'y2'],
            hyperpars_names=['tf']
        )
        self.num_parameters_to_save = 5
    
    def protocol_shape(self, pars):
        return make_bangramp_pulse_fun(pars, self.hyperpars['tf'])
    
    def fill_hyperpar_value(self, **kwargs):
        """Automatically add constraints whenever tf is filled."""
        for par_name, par_value in kwargs.items():
            if par_name == 'tf':
                self.add_parameter_constraints(t1=[0, par_value])
        super().fill_hyperpar_value(**kwargs)

    def constrain_intensities(self, boundaries):
        """Add constraints to all non-time parameters."""
        self.add_parameter_constraints(y0=boundaries,
                                       y1=boundaries, y2=boundaries)


class CRABProtocolAnsatz(ProtocolAnsatz):
    def __init__(self, num_frequencies):
        self.num_frequencies = num_frequencies
        pars_names = ['A' + str(idx + 1) for idx in range(num_frequencies)]
        pars_names += ['B' + str(idx + 1) for idx in range(num_frequencies)]
        nuk_names = ['nuk' + str(idx + 1) for idx in range(num_frequencies)]
        super().__init__(
            name='crab',
            pars_names=pars_names,
            hyperpars_names=['tf', 'y0', 'y1'] + nuk_names
        )
        logging.debug('Using {} frequencies'.format(num_frequencies))
        
        # option enabled by default
        self.generate_rnd_frequencies_each_tf = True
        # frequencies + amps A and B + y0 and y1
        self.num_parameters_to_save = 3 * num_frequencies + 3
    
    def protocol_shape(self, pars):
        nuk = [self.hyperpars['nuk' + str(idx + 1)]
               for idx in range(self.num_frequencies)]
        tf = self.hyperpars['tf']
        y0 = self.hyperpars['y0']
        y1 = self.hyperpars['y1']
        return make_CRAB_final_ramp_fun(pars, nuk, tf, y0, y1)

    def fill_hyperpar_value(self, **kwargs):
        """Automatically add constraints whenever tf is filled."""
        if self.generate_rnd_frequencies_each_tf and 'tf' in kwargs:
            new_freqs = np.random.uniform(
                low=-0.5, high=0.5, size=self.num_frequencies)
            for idx in range(self.num_frequencies):
                self.hyperpars['nuk' + str(idx + 1)] = new_freqs[idx]
            logging.debug('Regenerated CRAB frequencies: nuk={}'.format(
                new_freqs))
        super().fill_hyperpar_value(**kwargs)
    
    def constrain_all_amplitudes(self, boundaries):
        """Add given constraints to all A, B amplitudes."""
        constraints = {par_name: boundaries for par_name in self.pars_names}
        self.add_parameter_constraints(**constraints)


class TripleBangProtocolAnsatz(ProtocolAnsatz):
    def __init__(self):
        super().__init__(
            name='triplebang',
            pars_names=['y1', 'y2', 'y3', 't1', 't2'],
            hyperpars_names=['tf']
        )
        self.num_parameters_to_save = 6
    
    def protocol_shape(self, pars):
        y1, y2, y3, t1, t2 = pars
        tf = self.hyperpars['tf']
        def fun(t, *args):
            if 0 <= t <= t1:
                return y1
            elif t1 < t <= t2:
                return y2
            elif t2 < t <= tf:
                return y3
            else:
                return 200.
        return fun
    
    def fill_hyperpar_value(self, **kwargs):
        """Automatically add constraints whenever tf is filled."""
        for par_name, par_value in kwargs.items():
            if par_name == 'tf':
                self.add_parameter_constraints(t1=[0, par_value])
                self.add_parameter_constraints(t2=[0, par_value])
        super().fill_hyperpar_value(**kwargs)
    
    def constrain_intensities(self, boundaries):
        """Add constraints to all non-time parameters."""
        self.add_parameter_constraints(y1=boundaries, y2=boundaries,
                                       y3=boundaries)
    
    def are_pars_in_boundaries(self, pars):
        """Check that the parameters are consistent with each other.
        
        Used to ensure that t1 is less than t2
        """
        _, _, _, t1, t2 = pars
        if t1 > t2:
            return False
        return super().are_pars_in_boundaries(pars)