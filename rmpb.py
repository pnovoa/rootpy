import numpy as np
import mathutils
import itertools

class Environment:
    def __init__(self, dimension, num_peaks, initial_angle):
        self.H = np.zeros((dimension, num_peaks))
        self.W = np.zeros((dimension, num_peaks))
        self.C = np.zeros((dimension, num_peaks))
        self.S = 2 # initial angle
        self.timeStep = 0

class RMPB:
    def __init__(self):
        #CONSTANTS


        # search space boundaries
        self.X_MIN = -25.
        self.X_MAX = 25.

        #search space dimension
        self.DIM = 2

        # height
        self.H_MIN = 30.
        self.H_MAX = 70.
        self.H_SEV = 5.
        # width
        self.W_MIN = 1.
        self.W_MAX = 13.
        self.W_SEV = 0.5
        # angle
        self.S_MIN = -np.pi
        self.S_MAX = np.pi
        self.S_SEV = 1.0

        # initial angle for rotation
        self.INITIAL_ANGLE = 0.
        # chaotic constant
        self.A = 3.67
        # gamma
        self.GAMMA = 0.04
        # gamma max
        self.GAMMA_MAX = 0.1
        # period
        self.PERIOD = 12
        # noisy severity
        self.NOISY_SEV = 0.8

        self.RAND_SEED = 12345

        ## Factors subject of experimentation

        #number of peaks for each dimension
        self.num_peaks = 5
        #number of environment to learn
        self.learning_period = 20
        # time windows : number of future environments for R estimation
        self.time_windows = 2
        # number of function evaluations before a change
        self.computational_budget = 2500
        # number of changes for each simulation (run) : number of environments
        self.num_changes = 100
        # change type experimented for each simulation (run)
        self.change_type = 1

        ## Internal attributes

        # environments (corresponding to the initial and
        # those obtained after a change)
        self.environments = []

        self.curr_env = 0

        self.C_change = self.rotate_position
        self.P_change = self.ct_small_step

        self.ss = []

        self.minimize = False

    def init(self):

        self.rnd = np.random.RandomState(self.RAND_SEED)

        if(self.change_type == 1):
            self.P_change = self.ct_small_step
        elif(self.change_type == 2):
            self.P_change = self.ct_large_step
        elif(self.change_type == 3):
            self.P_change = self.ct_random
        elif(self.change_type == 4):
            self.P_change = self.ct_chaotic
            self.C_change = self.ct_dummy
        elif(self.change_type == 5):
            self.P_change = self.ct_recurrent
        elif(self.change_type == 6):
            self.P_change = self.ct_recurrent_with_noise

        self.curr_env = 0
        #self.rnd = np.random.RandomState(self.RAND_SEED)

        # initilizing the environments
        self.build_environments()

    def build_environments(self):
        self.environments = []
        self.ss = []
        #initial environment without change
        env0 = Environment(self.DIM, self.num_peaks, self.INITIAL_ANGLE)
        env0.C = self.rnd.uniform(low = self.X_MIN, high = self.X_MAX, size = (self.DIM, self.num_peaks))
        env0.H = self.rnd.uniform(low = self.H_MIN, high = self.H_MAX, size = (self.DIM, self.num_peaks))
        env0.W = self.rnd.uniform(low = self.W_MIN, high = self.W_MAX, size = (self.DIM, self.num_peaks))
        env0.timeStep = 0
        self.environments.append(env0)
        self.ss.append(env0.S)

        #generate the rest of the environments from env0

        for i in range(1, self.num_changes + self.time_windows):
            env = Environment(dimension=self.DIM, num_peaks=self.num_peaks, initial_angle=self.INITIAL_ANGLE)
            env.timeStep = i
            self.environments.append(env)
            self.P_change(i)
            self.C_change(i)
            self.ss.append(env.S)

    def evaluate(self, x):
        return self.eval_env(x, self.curr_env)

    def evaluate_vect(self, x):
        return np.apply_along_axis(self.evaluate, 1, x)

    def eval_env(self, x, env_id):
        env = self.environments[env_id]
        all_peaks = env.H - env.W * np.abs(env.C - np.tile(x, (self.num_peaks, 1)).transpose())
        max_peaks = np.max(all_peaks, axis=1)
        return np.mean(max_peaks)

    def true_robusteness_eval(self, x):
        result = [self.eval_env(x, env_id) for env_id in range(self.curr_env, self.curr_env + self.time_windows - 1)]
        return self.robustness_definition(result)

    def true_robusteness_eval_vect(self, x):
        return np.apply_along_axis(self.true_robusteness_eval, 1, x)

    def robustness_definition(self, vect_f):
        return np.mean(vect_f)

    def change(self):
        self.curr_env += 1

    def rotate_position(self, env_id):
        env = self.environments[env_id]
        prev_env = self.environments[env.timeStep-1]
        #Rotation matrix

        c, s = np.cos(env.S), np.sin(env.S)
        rot_mat = np.array(((c, -s), (s, c)))

        def apply_rotation(col_vect):
            return np.dot(col_vect, rot_mat)

        env.C = np.apply_along_axis(apply_rotation, 0, prev_env.C)
        env.C = np.clip(env.C, self.X_MIN, self.X_MAX)


    def ct_small_step(self, env_id):

        def change(prev_data, min_val, max_val, sev, gamma, rnd_val):
            result = prev_data + gamma * (max_val - min_val) * sev * (2* rnd_val - 1)
            return result.clip(min_val, max_val)

        env = self.environments[env_id]
        prev_env = self.environments[env.timeStep-1]

        env.H = change(prev_env.H, self.H_MIN, self.H_MAX, self.H_SEV, self.GAMMA, self.rnd.uniform(size=prev_env.H.shape))
        env.W = change(prev_env.W, self.W_MIN, self.W_MAX, self.W_SEV, self.GAMMA, self.rnd.uniform(size=prev_env.W.shape))
        env.S = change(np.array([prev_env.S]), self.S_MIN, self.S_MAX, self.S_SEV, self.GAMMA, self.rnd.uniform(size=(1,)))
        env.S = env.S[0]


    def ct_large_step(self, env_id):

        def change(prev_data, min_val, max_val, sev, gamma, rnd_val):

            result = 2 * rnd_val - 1
            result = prev_data + (max_val-min_val)*(gamma * mathutils.sign(result) + (self.GAMMA_MAX - gamma)* result) * sev
            return result.clip(min_val, max_val)

        env = self.environments[env_id]
        prev_env = self.environments[env.timeStep-1]

        env.H = change(prev_env.H, self.H_MIN, self.H_MAX, self.H_SEV, self.GAMMA, self.rnd.uniform(size=prev_env.H.shape))
        env.W = change(prev_env.W, self.W_MIN, self.W_MAX, self.W_SEV, self.GAMMA, self.rnd.uniform(size=prev_env.W.shape))
        env.S = change(np.array([prev_env.S]), self.S_MIN, self.S_MAX, self.S_SEV, self.GAMMA, self.rnd.uniform(size=(1,)))
        env.S = env.S[0]



    def ct_random(self, env_id):
        def change(prev_data, min_val, max_val, sev, rnd_val):

            result = prev_data * rnd_val * sev
            return result.clip(min_val, max_val)

        env = self.environments[env_id]
        prev_env = self.environments[env.timeStep-1]

        env.H = change(prev_env.H, self.H_MIN, self.H_MAX, self.H_SEV, self.rnd.normal(size=prev_env.H.shape))
        env.W = change(prev_env.W, self.W_MIN, self.W_MAX, self.W_SEV, self.rnd.normal(size=prev_env.W.shape))
        env.S = change(np.array([prev_env.S]), self.S_MIN, self.S_MAX, self.S_SEV, self.rnd.normal(size=(1,)))
        env.S = env.S[0]

    def ct_dummy(self, env_id):
        pass

    def ct_chaotic(self, env_id):
        def change(prev_data, min_val, max_val):
            result = min_val * self.A * (prev_data - min_val) * (1 - (prev_data - min_val)/(max_val-min_val))
            return result.clip(min_val, max_val)

        env = self.environments[env_id]
        prev_env = self.environments[env.timeStep-1]

        env.H = change(prev_env.H, self.H_MIN, self.H_MAX)
        env.W = change(prev_env.W, self.W_MIN, self.W_MAX)
        env.C = change(prev_env.C, self.X_MIN, self.X_MAX)
        #env.S = env.S[0]

    def ct_recurrent(self, env_id):
        def change(prev_data, min_val, max_val, angle):

            result = min_val + (max_val-min_val) *(np.sin(2*(np.pi*env_id)/self.PERIOD + angle) + 1)/2.;
            return result.clip(min_val, max_val)

        env = self.environments[env_id]
        prev_env = self.environments[env.timeStep-1]

        angles = np.array([x + y for x in range(self.DIM) for y in range(self.num_peaks)])
        angles = self.PERIOD * angles/(self.DIM + self.num_peaks)
        angles = np.reshape(angles, (self.DIM, self.num_peaks))

        env.H = change(prev_env.H, self.H_MIN, self.H_MAX, angles)
        env.W = change(prev_env.W, self.W_MIN, self.W_MAX, angles)
        env.S = 2*np.pi/self.PERIOD

    def ct_recurrent_with_noise(self, env_id):
        def change(prev_data, min_val, max_val, angle, rnd_val):
            result = min_val + (max_val-min_val) *(np.sin(2*(np.pi*env_id)/self.PERIOD + angle) + 1)/2.;
            result = result + self.NOISY_SEV*rnd_val
            return result.clip(min_val, max_val)

        env = self.environments[env_id]
        prev_env = self.environments[env.timeStep-1]

        angles = np.array([x + y for x in range(self.DIM) for y in range(self.num_peaks)])
        angles = self.PERIOD * angles/(self.DIM + self.num_peaks)
        angles = np.reshape(angles, (self.DIM, self.num_peaks))

        env.H = change(prev_env.H, self.H_MIN, self.H_MAX, angles, self.rnd.normal(size=prev_env.H.shape))
        env.W = change(prev_env.W, self.W_MIN, self.W_MAX, angles, self.rnd.normal(size=prev_env.W.shape))
        env.S = 2*np.pi/self.PERIOD
