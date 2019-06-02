from rmpb import RMPB

#from root_frmk import RootFramework
from jin_frmk import JinFramework

import numpy as np
from numpy import genfromtxt
from itertools import product

from matplotlib import pyplot as plt
from scipy import stats

from sklearn.metrics import mean_squared_error

import warnings

import sys

from multiprocessing import Pool

from scipy.optimize import differential_evolution as de_optimizer


#Bayesian optimization
#from sklearn.base import clone
#from skopt import gp_minimize
#from skopt.learning import GaussianProcessRegressor
#from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
#from bayesian_optimization_util import *


class RunSettings:
    def __init__(self, args, points):
        self.frm_cls = eval(args[0])
        self.frm_id = int(args[1])
        self.change_type = int(args[2])
        self.time_windows = int(args[3])
        self.learning_period = int(args[4])
        self.num_runs = int(args[5])
        self.num_changes = int(args[6])
        self.num_iter = int(args[7])
        self.pop_size = int(args[8])
        self.output_file_name = args[9]
        self.points = points
        self.samplesize = np.shape(points)[0]
        self.opt_seed = 3276
        self.opt_noise = 0.1

        #self.pso_options = {'c1': 1.496, 'c2': 1.496, 'w':0.729}
        #self.n_particles = np.shape(points)[1]*3


#def create_optimizer():



def perform_single_run(runid, runset):

    problem1 = RMPB()
    problem1.time_windows = runset.time_windows
    problem1.learning_period = runset.learning_period
    problem1.change_type = runset.change_type
    problem1.num_changes = runset.num_changes
    problem1.RAND_SEED += runid
    problem1.init()

    #myrandom  = np.random.RandomState(1245 + runid)

    data_x = problem1.X_MIN + (problem1.X_MAX-problem1.X_MIN) * runset.points
    # Build the framework
    frmw = runset.frm_cls(problem1, data_x)

    npoints = runset.points.shape[0]
    mshape = ((runset.num_changes - runset.learning_period), 10)

    perf_measures = np.zeros(mshape)
    perf_index = 0

    #bounds for optimizer
    x_max = problem1.X_MAX * np.ones(problem1.DIM)
    x_min = problem1.X_MIN * np.ones(problem1.DIM)
    limits = (problem1.X_MIN, problem1.X_MAX)
    ss_bounds = [limits]*problem1.DIM
    search_space_ext = np.linalg.norm(x_max-x_min)
    runset.opt_seed += runid

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for i in range(runset.num_changes):

            # Evaluate the sample points
            data_y = np.apply_along_axis(problem1.evaluate, 1, data_x)
            # And notify the framework about the change
            frmw.save_past(data_y)

            if(i >= problem1.learning_period):

                runset.opt_seed += i

                #data_y = frmw.eval_robustness_vect(data_x)

                #Scenario 1
                scenario1_res = de_optimizer(func=min_robustness, bounds=ss_bounds, args=(problem1.true_robusteness_eval,1), maxiter=runset.num_iter, popsize=runset.pop_size, seed=runset.opt_seed)
                scenario1_opt_f = -1*scenario1_res.fun
                scenario1_opt_x = scenario1_res.x

                #Scenario 2
                scenario2_res = de_optimizer(func=min_robustness, bounds=ss_bounds, args=(problem1.eval_robustness_single_knowing_past,1), maxiter=runset.num_iter, popsize=runset.pop_size, seed=runset.opt_seed)
                scenario2_opt_f = -1*scenario2_res.fun
                scenario2_opt_x = scenario2_res.x

                #Scenario 3
                scenario3_res = de_optimizer(func=min_robustness, bounds=ss_bounds, args=(frmw.eval_robustness_single,1), maxiter=runset.num_iter, popsize=runset.pop_size, seed=runset.opt_seed)
                scenario3_opt_f = -1*scenario3_res.fun
                scenario3_opt_x = scenario3_res.x


                scenario1_true_rob = scenario1_opt_f #problem1.true_robusteness_eval(scenario1_opt_x)
                scenario2_true_rob = problem1.true_robusteness_eval(scenario2_opt_x)
                scenario3_true_rob = problem1.true_robusteness_eval(scenario3_opt_x)

                perf_measures[perf_index, :] = [runset.frm_id, runset.samplesize, runset.change_type, runset.time_windows, runset.learning_period, runid, i, scenario1_true_rob, scenario2_true_rob, scenario3_true_rob]
                perf_index = perf_index + 1

            # A new change arrives...
            problem1.change()

    return perf_measures



def perform_experiment(args):

    #Parsing parameters
    nprocesses = int(args[0])
    samplesize = int(args[1])
    points = genfromtxt("points/points"+str(samplesize)+".csv", delimiter=",", skip_header=1)
    runset = RunSettings(args[2:], points)

    output_file_name = runset.output_file_name
    output_file_name += "_".join(args[1:-1]) + ".csv"
    f = open(output_file_name, "ab")

    runs = range(1, runset.num_runs + 1)

    #for nr in runs:
    #    res = perform_single_run(nr, runset)
    #    np.savetxt(f, res)

    with Pool(processes=nprocesses) as pool:
        result = pool.starmap(perform_single_run, product(runs, [runset]))
        for res in result:
            np.savetxt(f, res)

    print("Experiment {} finished".format(output_file_name))
    f.close()


def min_robustness(x, true_func, d):
    return -1*true_func(x)


def test_main():
    #Test problem evaluate
    problem1 = RMPB()
    problem1.time_windows = 3
    problem1.learning_period = 20
    problem1.change_type = 1
    problem1.num_changes = 23
    problem1.init()

    #pso_options = {'c1': 0.5, 'c2': 0.5, 'w':0.9}
    x_max = problem1.X_MAX
    x_min = problem1.X_MIN
    ss_bounds = np.array([[x_min, x_max]]*problem1.DIM)

    points = genfromtxt("points/points30.csv", delimiter=",", skip_header=1)
    data_x = problem1.X_MIN + (problem1.X_MAX - problem1.X_MIN) * points
    frm = JinFramework(problem1, data_x)

    

if __name__ == '__main__':
    perform_experiment(sys.argv[1:])
    #test_main()
