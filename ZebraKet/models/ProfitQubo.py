from math import log2, floor
from dimod import DiscreteQuadraticModel
import numpy as np
from models.AbstractQubo import AbstractQubo

class ProfitQubo(AbstractQubo):
    def __init__(self, profits=None, costs=None, budget: float=100, max_number_of_products=10) -> None:
        """Initializes the ProfitQubo
        
        profits (list):
            List of profits associated with the items
            If None, you must call self.build() before calling self.solve()
        costs (list):
            List of costs associated with the items
            If None, you must call self.build() before calling self.solve()
        budget (int):
            Maximum allowable cost
        max_number_of_products(int):
            Maximum allowable products for each product
        """
        super().__init__()
        self.profits = profits
        self.costs = costs
        self.budget = budget
        self.max_number_of_products = max_number_of_products + 1 # also take into account the value 0
        if profits is not None and costs is not None: 
            self.build(profits, costs)

    def build(self, profits: list[float], costs: list[float]): 
        """Bulds the qubo

        Args: 
        profits - list of profits we can get per product
        costs - list of costs per product
        """
        print(f'Building QUBO')
        self.profits = np.array(profits)
        self.costs = np.array(costs)
        self.qubo = self.construct_dqm()

    def construct_dqm(self): 

        values = self.profits
        weights = self.costs
        variable_bounds = list(np.random.randint(10, 50,size=(len(values))))
        weight_capacity = self.budget
        bound = [b+1 for b in variable_bounds] # also take into account the value 0
    
        # First guess the lagrange
        lagrange = max(values)*0.5
        
        # Number of objects
        x_size = len(values)

        # Lucas's algorithm introduces additional slack variables to
        # handle the inequality. M+1 binary slack variables are needed to
        # represent the sum using a set of powers of 2.
        M = floor(log2(weight_capacity))
        num_slack_variables = M + 1

        # Slack variable list for Lucas's algorithm. The last variable has
        # a special value because it terminates the sequence.
        y = [2**n for n in range(M)]
        y.append(weight_capacity + 1 - 2**M)
        
        ##@  Discrete Quadratic Model @##
        dqm = DiscreteQuadraticModel()
        
        self.x = []
        #@ Add variables @##
        for k in range(x_size):
            self.x.append(dqm.add_variable(bound[k], label='x' + str(k)))

        for k in range(num_slack_variables):
            dqm.add_variable(2, label='y' + str(k)) # either 0 or 1

        ##@ Hamiltonian xi-xi terms ##
        for k in range(x_size):
            pieces = range(bound[k])
            dqm.set_linear('x' + str(k), lagrange * (weights[k]**2) * (np.array(pieces)**2) - values[k]*pieces)


        # # Hamiltonian xi-xj terms
        for i in range(x_size):
            for j in range(i + 1, x_size):
                biases_dict = {}
                for piece1 in range(bound[i]):
                    for piece2 in range(bound[j]):
                        biases_dict[(piece1, piece2)]=(2 * lagrange * weights[i] * weights[j])*piece1*piece2

                dqm.set_quadratic('x' + str(i), 'x' + str(j), biases_dict)

        # Hamiltonian y-y terms
        for k in range(num_slack_variables):
            dqm.set_linear('y' + str(k), lagrange*np.array([0,1])* (y[k]**2))

        # Hamiltonian yi-yj terms 
        for i in range(num_slack_variables):
            for j in range(i + 1, num_slack_variables): 
                dqm.set_quadratic('y' + str(i), 'y' + str(j), {(1,1):2 * lagrange * y[i] * y[j]})

        # Hamiltonian x-y terms
        for i in range(x_size):
            for j in range(num_slack_variables):
                biases_dict = {}
                for piece1 in range(bound[i]):
                    biases_dict[(piece1, 1)]=-2 * lagrange * weights[i] * y[j]*piece1

                dqm.set_quadratic('x' + str(i), 'y' + str(j), biases_dict) 
    
        return dqm

    def construct_dqm_old(self):

        values = self.profits
        weights = self.costs
        weight_capacity = self.budget*100

        variable_bounds = list(np.random.randint(10, 50,size=(len(values))))
        bound = [b+1 for b in variable_bounds] # also take into account the value 
    
        # First guess the lagrange
        lagrange = max(values)*0.5
        
        # Number of objects
        x_size = len(values)

        # Lucas's algorithm introduces additional slack variables to
        # handle the inequality. M+1 binary slack variables are needed to
        # represent the sum using a set of powers of 2.
        M = floor(log2(weight_capacity))
        num_slack_variables = M + 1

        # Slack variable list for Lucas's algorithm. The last variable has
        # a special value because it terminates the sequence.
        y = [2**n for n in range(M)]
        y.append(weight_capacity + 1 - 2**M)
        
        ##@  Discrete Quadratic Model @##
        dqm = DiscreteQuadraticModel()
        
        x = []
        #@ Add variables @##
        for k in range(x_size):
            x.append(dqm.add_variable(bound[k], label='x' + str(k)))

        for k in range(num_slack_variables):
            dqm.add_variable(2, label='y' + str(k)) # either 0 or 1

        ##@ Hamiltonian xi-xi terms ##
        for k in range(x_size):
            pieces = range(bound[k])
            dqm.set_linear('x' + str(k), lagrange * (weights[k]**2) * (np.array(pieces)**2) - values[k]*pieces)


        # # Hamiltonian xi-xj terms
        for i in range(x_size):
            for j in range(i + 1, x_size):
                biases_dict = {}
                for piece1 in range(bound[i]):
                    for piece2 in range(bound[j]):
                        biases_dict[(piece1, piece2)]=(2 * lagrange * weights[i] * weights[j])*piece1*piece2

                dqm.set_quadratic('x' + str(i), 'x' + str(j), biases_dict)

        # Hamiltonian y-y terms
        for k in range(num_slack_variables):
            dqm.set_linear('y' + str(k), lagrange*np.array([0,1])* (y[k]**2))

        # Hamiltonian yi-yj terms 
        for i in range(num_slack_variables):
            for j in range(i + 1, num_slack_variables): 
                dqm.set_quadratic('y' + str(i), 'y' + str(j), {(1,1):2 * lagrange * y[i] * y[j]})

        # Hamiltonian x-y terms
        for i in range(x_size):
            for j in range(num_slack_variables):
                biases_dict = {}
                for piece1 in range(bound[i]):
                    biases_dict[(piece1, 1)]=-2 * lagrange * weights[i] * y[j]*piece1

                dqm.set_quadratic('x' + str(i), 'y' + str(j), biases_dict) 
        

        return dqm
    
    
if __name__ == "__main__":

    from dwave.system import LeapHybridDQMSampler
    from neal import SimulatedAnnealingSampler
    from config import standard_mock_data
    from utils.data import read_profit_optimization_data

    profits, costs = read_profit_optimization_data(standard_mock_data['small'])
    
    # qubo = ProfitQubo(LeapHybridDQMSampler().sample_dqm, profits=profits, costs=costs, budget=100, max_number_of_products=20)
    # qubo.solve()

    # sampler = SimulatedAnnealingSampler().sample_dqm

    sampler = LeapHybridDQMSampler().sample_dqm

    qubo = ProfitQubo(profits=profits, costs=costs, budget=300, max_number_of_products=20)
    qubo.solve(sampler)
    print(qubo.response)

    best_solution = qubo.response.first.sample    
    best_solution = [best_solution[i] for i in qubo.x]
    total_costs = sum([costs[index]*count for index, count in enumerate(best_solution)])
    total_profit = sum([profits[index]*count for index, count in enumerate(best_solution)])

    print('Total cost: ', total_costs)
    print('Total profit: ', total_profit)
    print('Best solution: ', best_solution)

