class RootFramework:

    '''
        @problem:       is the RMPB instance
        @sample_points: are the initial sample points
                        which is a matrix of m x dimension of the search
                        space
        @past_size:     is the number of past environments to include
                        for future prediction

        @past_approx_model: TODOOOOOOOOOOOOO

        @future_predict_model: TODOOOOOOO

    '''
    def __init__(self, problem, sample_points):
        self.problem = problem
        self.sample_points = sample_points
        self.past_size = problem.learning_period
        #print("and also here")
    '''
        It should use the information given by data_fitness, corresponding
        to the fitness (function evaluations) of sample_points, in order
        to save the past through a certain strategy. For instance,
        using an approximation model or an storage approach.
        It will be called after a change ocurrs in the problem.
    '''
    def save_past(self, data_fitness):
        pass



    '''
        It computes the estimated robustness for each xi in x. So, it return
        a vector with the robustness at current time step.
    '''
    def eval_robustness(self, x):
        pass


    '''
        It computes the estimated robustness for each xi in x (matrix). So, it return
        a vector with the robustness at current time step.
    '''
    def eval_robustness_vect(self, x):
        pass


    '''
        It computes the estimated robustness for each xi in x (1D vector). So, it return
        a vector with the robustness at current time step.
    '''
    def eval_robustness_single(self, x):
        pass
