import datetime
import logging
import os
import pickle
import functools
import sys
import numbers
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import qutip
#QuTiP control modules
import qutip.control.pulseoptim as cpo
# import qutip.logging_utils as logging
import scipy
import seaborn as sns

import src.rabi_model as rabi_model
import src.lmg_model as lmg_model
from src.utils import ground_state


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
    random_frequencies = 2 * np.pi * np.arange(1, N + 1) / N * (1 + nuk) / tf

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


def _make_CRAB_ramp_fun(Ak, Bk, nuk, tf, y0, y1):
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
    CRAB_correction = _make_CRAB_pulse_correction_fun(Ak, Bk, nuk, tf)
    def final_ramp_fun(t, *args):
        return CRAB_correction(t) * linear_segment(x0=0, x1=tf, y0=0, y1=y1, t=t)
    return final_ramp_fun


def make_CRAB_final_ramp_fun(parameters, nuk, tf, y0, y1):
    """Return overall pulse function used for CRAB model.

    Parameters
    ----------
    parameters : 1D np.array
        Contains parameters A and B in the standard CRAB ansatz. It is
        assumed to have length 2 * n where n is the number of frequencies.
    """
    Ak, Bk = parameters.reshape((2, len(parameters) // 2))
    return _make_CRAB_ramp_fun(Ak, Bk, nuk, tf, y0, y1)


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
    return make_linear_interpolation_fun([
        (0, y0), (t1, y0), (1.01 * t1, y1), (tf, y2)])


def make_doublebang_pulse_fun(parameters, tf):
    """Return double-bang pulse function.

    For 0 <= t <= t1, constant height y0.
    For t >= t1, constant height y1.
    """
    y0, t1, y1 = parameters
    return make_linear_interpolation_fun([
        (0, y0), (t1, y0), (1.01 * t1, y1), (tf, y1)])


def evolve_state(hamiltonian, initial_state, time,
                 return_all_states=False, time_frames=40):
    """Evolve a qutip.Qobj ket state according to the given Hamiltonian.

    This is a wrapper around `qutip.mesolve`, to which `hamiltonian` and
    `initial_state` are directly fed.

    Parameters
    ----------
    hamiltonian : list of qutip objects
        The Hamiltonian specification in `qutip.mesolve` is the
        "function based" one, as per qutip.mesolve documentation. This means
        that `hamiltonian` is to be given as a list of constant Hamiltonians,
        each one pairs with a time-dependent (numeric) coefficient.
        The simplest example would be `hamiltonian = [H0, [H1, coeffFun]]`.
    initial_state : qutip.Qobj or string
        A string can be used to compute the initial state from the hamiltonian.
        The accepted string options are:
            'ground state'.
        If not a string, it is assumed to be a qutip.Qobj object representing
        the initial ket state to be evolved.
    time : float or list of floats
        If a single number, it is divided into a number of subintervals and
        the result used as input for `qutip.mesolve`. If a list of numbers,
        it is directly fed to `qutip.mesolve`.
    """
    try:
        time[0]
        times_list = time
    except (TypeError, IndexError):  # if not a list or tuple, we assume `time` is a single number
        times_list = np.linspace(0, time, time_frames)
    
    # `initial_state` can be a string, specifying how the initial state should
    # be computed using the Hamiltonian (e.g. the initial state should be
    # the ground state of the Hamiltonian).
    # NOTE: This assumes that the first element of `hamiltonian` is the
    #       time-independent term. This is often but not necessarily the case.
    if isinstance(initial_state, str) and initial_state == 'ground state':
        if isinstance(hamiltonian, qutip.Qobj):  # happens for time-ind Hamiltonians
            initial_state = hamiltonian.eigenstates(eigvals=1)[1][0]
        else:
            initial_state = hamiltonian[0].eigenstates(eigvals=1)[1][0]

    evolving_states = qutip.mesolve(hamiltonian, initial_state, times_list)
    if return_all_states:
        return evolving_states.states
    else:
        return evolving_states.states[-1]


def evolve_adiabatically(initial_hamiltonian, final_hamiltonian, time,
                         time_frames=40, return_all_states=False):
    """Evolve the gs of an Hamiltonian towards that of another."""
    delta_ham = final_hamiltonian - initial_hamiltonian
    def linear_ramp(t, *args):
        return t / time
    H = [initial_hamiltonian, [delta_ham, linear_ramp]]

    times_list = np.linspace(0, time, time_frames)
    initial_state = ground_state(initial_hamiltonian)
    return evolve_state(hamiltonian=H, initial_state=initial_state,
                            time=times_list,
                            return_all_states=return_all_states)


# def run_CRAB_optimization(
#         hamiltonian, psi_initial, psi_target,
#         y0=0, y1=None,
#         tf=None, nuk=None,
#         initial_pars=np.zeros(4), stfu=False,
#         optim_method='Nelder-Mead'):
#     """Find pulse to reach ground state at the critical point.

#     This is essentially a wrapper around scipy.optimize.minimize, which is what
#     eventually performs the actual optimization.
    
#     Parameters
#     ----------
#     hamiltonian : tuple
#         A tuple of the form (H0, H1), with H0 time-independent term of the
#         Hamiltonian and H1 the operator to be attached to the time-dependent pulse.
#     psi_initial : qutip.Qobj
#         The initial state to be evolved
#     psi_target : qutip.Qobj
#         What we are trying to bring the initial state to.
#     y0, y1 : floats
#         Fixed initial and final height of the pulse.
#     tf : float
#         Total evolution time. The CRAB optimisation will try to find
#         an optimal protocol implementing the target in this amoutn of time.
#     nuk : array of floats
#         Array of (usually random) frequencies, as per the CRAB protocol.
#         If not given they are randomly choosen.
#     stfu : bool
#         Shut the fuck up.
#     """
#     if tf is None:
#         tf = 1.
#     if nuk is None:
#         nuk = np.random.rand(len(initial_pars) // 2)
#         if not stfu:
#             print('Random frequencies: {}'.format(nuk))

#     # make function to optimise
#     def fidelity_vs_CRAB_pars(CRAB_pars):
#         Ak, Bk = CRAB_pars.reshape((2, len(CRAB_pars) // 2))
#         final_ramp = _make_CRAB_ramp_fun(Ak, Bk, nuk, tf, y0=y0, y1=y1)
#         H = [hamiltonian[0], [hamiltonian[1], final_ramp]]
#         times_list = np.linspace(0, tf, 50)
#         output_state = qutip.mesolve(H, psi_initial, tlist=times_list)
#         return 1 - qutip.fidelity(psi_target, output_state.states[-1])
    
#     # run the actual optimisation
#     result = scipy.optimize.minimize(fidelity_vs_CRAB_pars, initial_pars,
#                                      method=optim_method)
#     result.data = dict(nuk=nuk, initial_pars=initial_pars)
#     return result


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
        if hyperpars_names is None:
            hyperpars_names = []
        self.hyperpars_names = hyperpars_names
        self.filled_hyperpars = {name: False for name in hyperpars_names}

        self.hyperpars = dict()

    def fill_hyperpar_value(self, **kwargs):
        """Fill in hyperparameters.
        
        Each input key should match one of the names in self.hyperpars_names
        """
        for name, value in kwargs.items():
            try:
                if self.filled_hyperpars[name]:
                    raise ValueError('The value of {} was already given'.format(name))
            except KeyError:
                raise ValueError('{} is not a recognised hyperparameter name.'.format(name))
            # otherwise just use the value
            self.hyperpars[name] = value
            self.filled_hyperpars[name] = True


    def time_dependent_fun(self, parameters):
        """Provided all hyperparameters have been filled, generate pulse shape.

        The length of the input must match the length of self.pars_names, and
        all hyperparameters must have been filled prior to calling this.
        """
        for name, filled_hyperpar in self.filled_hyperpars.items():
            if not filled_hyperpar:
                raise ValueError('The hyperparameter {} has not been given yet.'.format(name))
    

    def are_pars_in_boundaries(self, pars):
        """Check whether the parameters are in the range they should be in.
        """
        return True


class DoubleBangProtocolAnsatz(ProtocolAnsatz):
    def __init__(self):
        super().__init__(
            name='doublebang',
            pars_names=['y0', 't1', 'y1'],
            hyperpars_names=['tf']
        )
    
    def time_dependent_fun(self, pars):
        super().time_dependent_fun(pars)

        tf = self.hyperpars['tf']
        # impose soft constraint on the intermediate time
        return make_doublebang_pulse_fun(pars, tf)
    
    def are_pars_in_boundaries(self, pars):
        if pars[0] < 0 or pars[1] > self.hyperpars['tf']:
            return False


class BangRampProtocolAnsatz(ProtocolAnsatz):
    def __init__(self):
        super().__init__(
            name='bangramp',
            pars_names=['y0', 't1', 'y1', 'y2'],
            hyperpars_names=['tf']
        )
    
    def time_dependent_fun(self, pars):
        super().time_dependent_fun(pars)

        tf = self.hyperpars['tf']
        # impose soft constraint on the intermediate time
        return make_bangramp_pulse_fun(pars, tf)

    def are_pars_in_boundaries(self, pars):
        if pars[0] < 0 or pars[1] > self.hyperpars['tf']:
            return False


def optimize_model_parameters(
        hamiltonians, initial_state, target_state,
        evolution_time, parametrized_model,
        initial_parameters,
        parameters_constraints=None,
        overall_pulse_constraints=None,
        optimization_method='Nelder-Mead',
        optimization_options=None,
        stfu=False
    ):
    """Optimize a model ansatz with respect to its parameters.

    Input and output states are fixed, and so is the parametrized form of the
    protocol. Optimize the protocol-defining parameters to maximize the
    fidelity between initial_state and target_state.

    The total evolution time is generally fixed through the 

    Parameters
    ----------
    hamiltonians : pair of qutip.Qobj hamiltonians
        Pair [H0, H1] where H0 is the time-independent component of the overall
        Hamiltonian (the free term), and H1 the time-independent component of
        the time-dependent part of the overall Hamiltonian (the interaction
        term).
        During the optimization, different time-dependent functions will be
        attached to H1, according to the user-specified model.
        NOTE: Clearly, this restricts the use of this function to only
              a specific class of QOC problems.
    initial_state : qutip.Qobj state
        For every protocol tried, this state is evolved through the
        corresponding time-dependent Hamiltonian and compared with the target.
    target_state : qutip.Qobj state
        As above: the output state is for every protocol compared with this
        one (via qutip.fidelity).
    evolution_time : float
        The total evolution time. This could technically be specified through
        the parametrized model, but it's just easier for now to have it as a
        separate argument.
    parametrized_model : function
        Function taking as input a numpy array, and giving as output a
        real-to-real function returning, for a particular protocol, the
        interaction value corresponding to a given time.
    initial_parameters : list of floats or 1D np.array
        The optimization will proceed from this initial value of the protocol
        parameters.
    parameters_constraints : list of tuples
        Each element of the list should be a pair of floats, representing min
        and max value that the corresponding model parameter should be allowed
        to get. The len of the list should match the number of parameters of
        the model `parametrized_model`.
        A value of None instead of the (min, max) pair is accepted, and used
        to represent that no constraint should be assumed for that parameter.
    parameters_regularization : dict
        Used to specify (optional) regularization on the parameters.
        For example, a value of the form {1: 'linear', 3: 'quadratic'} means
        that linear regularization should be applied to the first parameter
        and quadratic regularization should be applied to the third parameter.
    overall_pulse_constraints : tuple
        Used to penalize a pulse when at any point during the evolution it
        gives intensities outside the provided range.
    optimization_method : string
        Passed over to scipy.optimize.minimize
    optimization_options : dict
        Passed over to scipy.optimize.minimize as the `options` parameteres.
        It is to be used to specify method-specific options.
    stfu : bool
        STFU
    """
    BIG_BAD_VALUE = 200  # yes, I pulled this number out of my ass

    if parameters_constraints is None:
        parameters_constraints = [(None, None)] * len(initial_parameters)

    def fidelity_vs_model_parameters(pars):
        # assign very high cost if outside of the boundaries (we have to do
        # this because for some reason scipy does not support boundaries with
        # Nelder-Mead and Powell)
        for par, constraint in zip(pars, parameters_constraints):
            if ((constraint[0] is not None and par < constraint[0]) or
                    (constraint[1] is not None and par > constraint[1])):
                return BIG_BAD_VALUE
        # build model, hamiltonian, and compute output state
        time_dependent_model_fun = parametrized_model(pars)

        # If needed, check that the overall protocol is not out of line (mostly
        # used for the CRAB protocol).
        # We don't want to add too much overhead here, so we use a pretty rough
        # sampling procedure to check this
        if overall_pulse_constraints is not None:
            for time_sample in np.linspace(0, evolution_time, 20):
                sample = time_dependent_model_fun(time_sample)
                if (sample < overall_pulse_constraints[0] or
                        sample > overall_pulse_constraints[1]):
                    logging.debug('BAAAAAAAAAD BOY!')
                    return BIG_BAD_VALUE

        # 
        H = [hamiltonians[0], [hamiltonians[1], time_dependent_model_fun]]
        output_state = evolve_state(H, initial_state, evolution_time)
        # compute and return infidelity (because scipy gives you MINimize)
        return 1 - qutip.fidelity(output_state, target_state)

    # run the actual optimisation
    if not stfu:
        print('Starting optimization')

    logging.info('Starting optimization for tf={}'.format(evolution_time))
    logging.debug('Optimization method: {}'.format(optimization_method))
    logging.debug('Optimization options: {}'.format(optimization_options))
    logging.info('Initial parameter values: {}'.format(initial_parameters))
    logging.info('Constraints: {}'.format(parameters_constraints))
    logging.info('Overall pulse constraints: {}'.format(overall_pulse_constraints))
    result = scipy.optimize.minimize(
        fun=fidelity_vs_model_parameters,
        x0=initial_parameters,
        method=optimization_method,
        options=optimization_options
        # bounds=parameters_constraints  # not accepted for Powell and NM
    )
    result.data = dict(initial_pars=initial_parameters)
    return result


def _optimize_model_parameters_scan_times(
        times_to_try=None,
        parametrized_protocol=None,
        hamiltonians=None,
        initial_state=None, target_state=None,
        initial_parameters=None,
        optimization_method=None,
        stfu=True,
        parameters_constraints=None,
        overall_pulse_constraints=None):
    """Run a series of OC optimizations for different times.

    Parameters
    ----------
    initial_parameters : list
        List of the same length of the number of parameters defining the
        protocol (total time excluded).
        Each element is either a number or a pair of two numbers. If a number,
        this value is used for all optimizations as initial value for the
        corresponding parameter.
        If a pair of numbers, at every iteration a random value in the given
        range is used as initial value for the corresponding parameter.
    hamiltonians : pair of qutip.Qobj hamiltonians
        Passed to optimize_model_parameters, see docs there.
    parametrized_protocol : function
        A function with prototype `fun(list_of_pars, tf)`, with `tf` a float
        representing the total evolution time of the protocol. The name of the
        second parameter is important because it is used for functools.partial.
    parameters_constraints : list
        Mostly passed over and parsed by optimize_model_parameters, but  some
        special values might be parsed here. In particular, elements with the
        string 'timeframe' are replaced at each stage with the pair `[0, tf]`,
        where `tf` is the total evolution time at the given iteration.
    """
    # parse initial parameters
    parsed_initial_pars = []
    for par in initial_parameters:
        if isinstance(par, numbers.Number):
            parsed_initial_pars.append(par)
        elif isinstance(par, str):
            # strings can be used to indicated special values. Only some values
            # are accepted
            if par != 'halftime':
                raise ValueError('Unrecognised value for initial parameters:'
                                 ' {}'.format(initial_parameters))
            parsed_initial_pars.append(par)
        else:  # we otherwise assume a pair (min, max) and generate rnd values
            min_, max_ = par
            random_value = np.random.rand() * (max_ - min_) + min_
            parsed_initial_pars.append(random_value)

    # start main iteration of optimizations
    results = np.zeros(shape=[len(times_to_try), len(initial_parameters) + 2])
    for idx, tf in enumerate(times_to_try):
        # parsed_initial_pars might still contain stuff to be computed at each
        # iteration. Parse that shit.
        initial_pars = np.zeros(len(parsed_initial_pars))
        for par_idx, par in enumerate(parsed_initial_pars):
            if not isinstance(par, numbers.Number):
                if isinstance(par, str) and par == 'halftime':
                    initial_pars[par_idx] = tf / 2.
                elif isinstance(par, (list, tuple)):
                    initial_pars[par_idx] = np.random.uniform(par[0], par[1])
            else:  # otherwise just use the fucking number
                initial_pars[par_idx] = par

        # parse tf-dependent parameter constraints
        parsed_parameters_constraints = parameters_constraints[:]
        for par_idx, constraints in enumerate(parsed_parameters_constraints):
            if isinstance(constraints, str) and constraints == 'timeframe':
                parsed_parameters_constraints[par_idx] = [0., tf]
        #
        parametrized_protocol = functools.partial(parametrized_protocol, tf=tf)
        result = optimize_model_parameters(
            hamiltonians=hamiltonians,
            initial_state=initial_state, target_state=target_state,
            evolution_time=tf,
            parametrized_model=parametrized_protocol,
            initial_parameters=initial_pars,
            optimization_method=optimization_method,
            parameters_constraints=parsed_parameters_constraints,
            overall_pulse_constraints=overall_pulse_constraints,
            stfu=stfu
        )
        fidelity = (1 - result.fun)**2
        logging.info('Fidelity: {}'.format(fidelity))
        results[idx] = [tf, fidelity, *result.x]

    pars_cols_names = ['par' + str(idx + 1) for idx in
                       range(len(initial_parameters))]
    results = pd.DataFrame(results, columns=['tf', 'fid', *pars_cols_names])
    return results


def find_best_protocol(
        problem_specification, optimization_specs,
        other_options={}
    ):
    """Higher-level interface to optimize_model_parameters.
    
    Parameters
    ----------
    problem_specification : dict
        Mandatory keys: 'model', 'model_parameters', 'task', 'time'.
        The accepted values for the `model` key are
            - 'rabi'
            - 'lmg'

        If model='rabi', the accepted values for `model_parameters` are
            - 'N'
            - 'Omega'
            - 'omega_0'
        If model='lmg', the accepted values for `model_parameters` are
            - 'num_spins'
        
        The accepted values for `task` are:
            - 'critical point state generation'

        The value of 'time' should be a positive number, representing the
        evolution time.

    optimization_specs : dict
        Accepted keys:
            - 'protocol'
            - 'protocol_options'
            - 'optimization_method'
            - 'optimization_options'
            - 'initial_parameters'

        The accepted values for `protocol` are
            - 'doublebang'
            - 'bangramp'
            - 'crab'
        The accepted values for 'protocol_options' depend on 'protocol'.
        If protocol='crab' then the accepted values are
            - 'num_frequencies'
            - 'OIJSODJASODJOIASJDOISAJDOIASJDOIASJDMOSANDOIUASNFAS'

        The accepted values for 'optimization_method' are those accepted by
        scipy.optim.minimize, and similarly the value of 'optimization_options'
        is passed over to this function.
        Other values accepted for optimization_specs are:
            - 'parameters_constraints'

        The value of 'initial_parameters' can be either a numpy array with
        explicit values, or a string.

    """
    model = problem_specification['model']
    model_parameters = problem_specification['model_parameters']
    task = problem_specification['task']
    protocol = optimization_specs['protocol']
    optim_method = optimization_specs['optimization_method']

    H0 = None
    H1 = None
    initial_state = None
    target_state = None

    if model == 'rabi':
        N = model_parameters['N']
        Omega = model_parameters['Omega']
        omega_0 = model_parameters['omega_0']
        if task == 'critical point state generation':
            critical_val = np.sqrt(Omega * omega_0) / 2.
            # build initial and final Hamiltonians
            H0 = rabi_model.QRM_free_term(N, Omega, omega_0)
            H1 = rabi_model.QRM_interaction_term(N)
            initial_state = ground_state(H0)
            target_state = ground_state(H0 + critical_val * H1)
        else:
            raise ValueError('Unrecognised task')
    elif model == 'lmg':
        num_spins = model_parameters['num_spins']
        if task == 'critical point state generation':
            critical_val = 1.  # this value follows from how the model definition
            # build initial and final hamiltonians
            # We assume to operate in the sector with maximal AM, which is
            # why we use `total_j=None` here
            H0 = lmg_model.LMG_free_term(num_spins, total_j=None)
            H1 = lmg_model.LMG_interaction_term(num_spins, g_value=critical_val,
                                                total_j=None)
            initial_state = ground_state(H0)
            target_state = ground_state(H0 + H1)
        else:
            raise ValueError('Unrecognised task')
    else:
        raise ValueError('{} is not a valid value for the'
                         'model.'.format(model))

    # determine protocol ansatz to use
    if protocol == 'doublebang':
        logging.info('Using doublebang protocol ansatz.')
        protocol_ansatz_fun = make_doublebang_pulse_fun
    elif protocol == 'bangramp':
        logging.info('Using bangramp protocol ansatz.')
        protocol_ansatz_fun = make_bangramp_pulse_fun
    elif protocol == 'crab':
        logging.info('Using CRAB protocol ansatz.')

        protocol_options = optimization_specs.get('protocol_options', {})
        CRAB_options = dict()
        # determine number of frequencies to use with CRAB protocol
        if 'num_frequencies' not in protocol_options:
            # default value kind of picked at random
            CRAB_options['num_frequencies'] = 2
            logging.info('(CRAB) Default value of {} frequencies chosen for th'
                         'e optimization'.format(
                             CRAB_options['num_frequencies']))
        else:
            CRAB_options['num_frequencies'] = protocol_options['num_frequencies']
        logging.info('(CRAB) Using {} frequencies.'.format(
            CRAB_options['num_frequencies']))

        # determine values of frequencies
        if 'frequencies' not in protocol_options:
            CRAB_options['frequencies'] = [-0.5, 0.5]
        else:
            CRAB_options['frequencies'] = protocol_options['frequencies']

        # fix starting and final points for pulse protocol in the case of
        # critical state generation task
        if task == 'critical point state generation':
            protocol_ansatz_fun = functools.partial(
                make_CRAB_final_ramp_fun,
                y0=0., y1=critical_val
            )
    else:
        raise ValueError('Unrecognised value of `protocol`.')

    # run the actual optimization
    if task == 'critical point state generation':

        # the scan_times option indicates that we want to perform a series of
        # optimizations for various values of the time parameter
        if 'scan_times' in other_options:

            # in the case of the CRAB protocol, we still need to decide the
            # (generally random) frequencies to use
            if protocol == 'crab':
                min_, max_ = CRAB_options['frequencies']
                nuk = np.random.uniform(low=min_, high=max_,
                                        size=CRAB_options['num_frequencies'])
                logging.debug('Generated new random CRAB frequencies'
                              ': {}'.format(nuk))
                protocol_ansatz_fun = functools.partial(
                    protocol_ansatz_fun, nuk=nuk)

            # parse initial parameters
            if 'initial_parameters' not in optimization_specs:
                # if not explicitly given, use as default double the critical
                # point for intensities and the timeframe for times
                if protocol == 'doublebang':
                    init_pars = [[0., 2 * critical_val], 'halftime',
                                 [0., 2 * critical_val]]
                elif protocol == 'bangramp':
                    init_pars = [[0., 2 * critical_val], 'halftime',
                                 [0., 2 * critical_val],
                                 [0., 2 * critical_val]]
                elif protocol == 'crab':
                    # I don't know, let's just try with amplitudes randomly
                    # sampled in the [-1, 1] interval (why not right?)
                    init_pars = [[-1, 1]] * (2 * CRAB_options['num_frequencies'])
            else:
                # if explicitly given, just assume the values make sense for
                # _optimize_model_parameters_scan_times
                init_pars = optimization_specs['initial_parameters']

            # parse parameters constraints
            if 'parameters_constraints' not in optimization_specs:
                # if not explicitly given, we generate boundaries similar to
                # those generated for default initial parameters
                overall_pulse_constraints = None
                critical_range = [-2 * critical_val, 2 * critical_val]
                if protocol == 'doublebang':
                    pars_constraints = [critical_range, 'timeframe',
                                        critical_range]
                elif protocol == 'bangramp':
                    pars_constraints = [critical_range, 'timeframe',
                                        critical_range, critical_range]
                elif protocol == 'crab':
                    # not sure if this will work well in general, but in this
                    # case we do not impose constraints on the parameters,
                    # but rather only on the overall pulse, via the appropriate
                    # option accepted by _optimize_model_parameters_scan_times
                    pars_constraints = [[None, None]] * (2 * CRAB_options['num_frequencies'])
                    overall_pulse_constraints = critical_range

            # run optimization
            results = _optimize_model_parameters_scan_times(
                times_to_try=other_options['scan_times'],
                parametrized_protocol=protocol_ansatz_fun,
                hamiltonians=[H0, H1],
                initial_state=initial_state, target_state=target_state,
                initial_parameters=init_pars,
                parameters_constraints=pars_constraints,
                overall_pulse_constraints=overall_pulse_constraints,
                optimization_method=optim_method
            )
            return results
        
        else:  # only scan_times works atm
            raise NotImplementedError('Only scan_times works atm, sorry.')
    

