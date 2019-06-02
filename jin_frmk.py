from root_frmk import *
from rmpb import RMPB
import numpy as np
from numpy import genfromtxt

from scipy import stats

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error

import pmdarima as pm

import warnings
import timeit


class JinFramework(RootFramework):

    def __init__(self, problem, sample_points, rnd_seed=124):
        super().__init__(problem, sample_points)
        self.past_approx_models = list()
        self.rnd = np.random.RandomState(rnd_seed)



    # Save the past information corresponding to sample_points
    def save_past(self, data_fitness):

        # We build a new approx. model and add it
        # the past_approx_models list (queue)
        kernel = RBF(length_scale=(1.,1.))
        am = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=self.rnd)
        am.fit(self.sample_points, data_fitness)

        if(len(self.past_approx_models) > self.problem.learning_period):
            self.past_approx_models.pop(0) # Remove the first element of the queue
        #Add the approx. model to the list
        self.past_approx_models.append(am)

    def eval_single_sol_past_fitness(self, x):
        result = np.array([am.predict(np.array([x]), return_std=False) for am in self.past_approx_models])
        return np.ndarray.flatten(result)

    def eval_single_sol_knowing_past_fitness(self, x):
        result = np.array([self.problem.eval_env(x, self.problem.curr_env-i) for i in range(1, self.problem.learning_period + 1)])
        return np.ndarray.flatten(result)

    def future_forecast(self, time_series, steps):
        arfit = pm.auto_arima(time_series, error_action='ignore', suppress_warnings=True, method="nm")
        return arfit.predict(n_periods=steps)


    def eval_robustness_single_knowing_past(self, x):
        present_fitness = self.problem.evaluate(x)
        past_fitness = self.eval_single_sol_knowing_past_fitness(x)
        past_present_fitness = np.append(past_fitness, present_fitness)
        future_fitness = self.future_forecast(past_present_fitness, self.problem.time_windows - 1)
        present_future_fitness = np.append(present_fitness, future_fitness)
        est_robust = np.mean(present_future_fitness)
        return est_robust

    def eval_robustness_single(self, x):

        # Evaluate x in the present environment
        present_fitness = self.problem.evaluate(x)
        #print("present_fitness:")
        #print(present_fitness)

        # Evaluate each xi in x, in the past environments
        past_fitness = self.eval_single_sol_past_fitness(x)
        #print("past_fitness:")
        #print(past_fitness)

        # Concat the past and present environments
        past_present_fitness = np.append(past_fitness, present_fitness)
        #print("past_present_fitness:")
        #print(past_present_fitness)

        # Forecast x using the past and present environments
        #startt = timeit.default_timer()
        future_fitness = self.future_forecast(past_present_fitness, self.problem.time_windows - 1)
        #print("future_fitness:")
        #print(future_fitness)
        #endt = timeit.default_timer()
        #print("This is the time of future_forecast {}".format(endt - startt))

        # Concat present and future environments for Robustness computation
        present_future_fitness = np.append(present_fitness, future_fitness)
        #print("present_future_fitness:")
        #print(present_future_fitness)

        # Compute Robustness for each xi in x # # TODO: Call the robustness_definition at rmpb
        est_robust = np.mean(present_future_fitness)

        return est_robust



    def eval_robustness_vect(self, x):
        return np.apply_along_axis(self.eval_robustness_single, 1, x)

    def eval_robustness(self, x):

        # Evaluate x in the present environment
        present_fitness = np.apply_along_axis(self.problem.evaluate, 1, x)
        #print("present_fitness:")
        #print(present_fitness)

        # Evaluate each xi in x, in the past environments
        past_fitness = np.apply_along_axis(self.eval_single_sol_past_fitness, 1, x)
        #print("past_fitness:")
        #print(past_fitness)

        # Concat the past and present environments
        past_present_fitness = np.concatenate((past_fitness, np.array([present_fitness]).T), axis=1)
        #print("past_present_fitness:")
        #print(past_present_fitness)

        # Forecast x using the past and present environments
        #startt = timeit.default_timer()
        future_fitness = np.apply_along_axis(self.future_forecast, 1, past_present_fitness, self.problem.time_windows - 1)
        #print("future_fitness:")
        #print(future_fitness)
        #endt = timeit.default_timer()
        #print("This is the time of future_forecast {}".format(endt - startt))

        # Concat present and future environments for Robustness computation
        present_future_fitness = np.concatenate((np.array([present_fitness]).T, future_fitness), axis=1)
        #print("present_future_fitness:")
        #print(present_future_fitness)

        # Compute Robustness for each xi in x # # TODO: Call the robustness_definition at rmpb
        est_robust = np.mean(present_future_fitness, axis=1)

        return est_robust



if __name__ == '__main__':
    problem = RMPB()
    problem.time_windows = 3
    problem.init()
    sample_points = np.random.uniform(-25, 25, (50,2))
    test_points = np.random.uniform(-25, 25, (40,2))

    jin_frmk = JinFramework(problem, sample_points)

    for chg in range(problem.learning_period + 3):
        sample_fitness = np.apply_along_axis(problem.evaluate, 1, sample_points)
        jin_frmk.save_past(sample_fitness)
        if(chg >= problem.learning_period):
            print(jin_frmk.eval_robustness(test_points))
        problem.change()
