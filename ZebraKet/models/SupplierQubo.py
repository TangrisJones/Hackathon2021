from dimod import BinaryQuadraticModel, BINARY
import numpy as np
from models.AbstractQubo import AbstractQubo

class SupplierQubo(AbstractQubo):
    # Lagrange multipliers A>B>0
    lagrange_a = 2
    lagrange_b = 1

    def __init__(self, inventory: list[int or str] or None, supplier_inventory: list[set[int or str]] or None) -> None:
        """Initializes the SupplierQubo
        
        inventory (list):
            List of items we want for our inventory
        supplier_inventory (list of sets):
            List for each supplier their inventory
        """
        super().__init__()
        self.inventory = None
        self.supplier_inventory = None
        self.indicators = None

        if inventory is not None and supplier_inventory is not None:
            self.build(inventory, supplier_inventory)

    def build(self, inventory: list[int or str] or None, supplier_inventory: list[set[int or str]] or None): 
        """Bulds the qubo

        Args: 
        inventory (list):
            List of items we want for our inventory
        supplier_inventory (list of sets):
            List for each supplier their inventory
        """
        print(f'Building QUBO')
        # profits: list[float], costs: list[float], budget: float, max_number_of_products=10
        self.inventory = inventory
        self.supplier_inventory = supplier_inventory
        self.qubo = self.construct_bqm()

    def print_best(self):
        best_solution = self.response.first.sample
        best_solution = [best_solution[i] for i in self.x]
        print(f'Found {np.sum(best_solution)} suppliers with energy {self.energy_set[0]}')
        print('All solutions: ', self.solution_set)
        print('\n')
        

    def construct_bqm(self):
        """Construct BQM for the set cover problem
        Args:
            U (array-like):
                A set of elements defining the universe
            V (array of sets):
                Array of subsets
        Returns:
            Binary quadratic model instance
            x: variable
        """

        U = self.inventory
        V = self.supplier_inventory
        A = self.lagrange_a
        B = self.lagrange_b
        
        # Create indicator variables
        self.indicators = []
        for i in range(len(V)):
            self.indicators.append([1 if U[a] in V[i] else 0 for a in range(len(U))])
        
        ##@  Binary Quadratic Model @##
        bqm = BinaryQuadraticModel('BINARY')

        # Add linear terms
        # x linear terms
        self.x = [bqm.add_variable('x_'+str(i+1), A*sum(self.indicators[i])+B) for i in range(0,len(V))]
        
        # y_am linear terms
        y = []
        for a in range(1,len(U)+1):
            y.append([bqm.add_variable('y_('+str(a)+', '+str(m)+')', A*(m**2-1)) for m in range(1,len(V)+1)])
        
        # Add quadratic terms

        # x_i-x_j terms
        for i in range(1,len(V)+1):
            for j in range(i+1,len(V)+1):
                key = ('x_' + str(i), 'x_' + str(j))
                bqm.quadratic[key] = 2*A*np.dot(np.array(self.indicators[i-1]),np.array(self.indicators[j-1]))

        # y_am - y_an terms
        for m in range(1,len(V)+1):
            for n in range(m+1,len(V)+1):
                for a in range(1,len(U)+1):
                    key = ('y_('+str(a)+', '+str(m)+')', 'y_('+str(a)+', '+str(n)+')')
                    bqm.quadratic[key] = 2*A*(1+m*n)

        # x_i-y_am terms
        for i in range(1,len(V)+1):
            for m in range(1,len(V)+1):
                for a in range(1,len(U)+1):
                    key = ('x_' + str(i), 'y_('+str(a)+', '+str(m)+')')
                    bqm.quadratic[key] = -2*A*m*self.indicators[i-1][a-1]

        return bqm

    def construct_bqm_old(self):
        """Construct BQM for the generalized set cover problem
        Args:
            
        Returns:
            Binary quadratic model instance
        """

        # Create indicator variables
        self.indicators = []
        for i in range(len(self.supplier_inventory)):
            self.indicators.append([1 if self.inventory[a] in self.supplier_inventory[i] else 0 for a in range(len(self.inventory))])

        ##@  Binary Quadratic Model @##
        bqm = BinaryQuadraticModel(BINARY)

        # Add linear terms
        # x linear terms
        self.x = [bqm.add_variable(f'x_{i+1}', SupplierQubo.lagrange_a*sum(self.indicators[i])+SupplierQubo.lagrange_b) for i in range(0,len(self.supplier_inventory))]
        # print('x variables:',x)

        # y_am linear terms
        y = []
        for a in range(1,len(self.inventory)+1):
            y.append([bqm.add_variable(f'y_{a,m}', SupplierQubo.lagrange_a*(m**2-1)) for m in range(1,len(self.supplier_inventory)+1)])
        # print('y variables:',y)

        # Add quadratic terms

        # x_i-x_j terms
        for i in range(1,len(self.supplier_inventory)+1):
            for j in range(i+1,len(self.supplier_inventory)+1):
                key = ('x_' + str(i), 'x_' + str(j))
                bqm.quadratic[key] = 2*SupplierQubo.lagrange_a*np.dot(np.array(self.indicators[i-1]),np.array(self.indicators[j-1]))
                
        # y_am - y_an terms
        for m in range(1,len(self.supplier_inventory)+1):
            for n in range(m+1,len(self.supplier_inventory)+1):
                for a in range(1,len(self.inventory)+1):
                    key = ('y_('+str(a)+', '+str(m)+')', 'y_('+str(a)+', '+str(n)+')')
                    bqm.quadratic[key] = 2*SupplierQubo.lagrange_a*(1+m*n)
                    
        # x_i-y_am terms
        for i in range(1,len(self.supplier_inventory)+1):
            for m in range(1,len(self.supplier_inventory)+1):
                for a in range(1,len(self.inventory)+1):
                    key = ('x_' + str(i), 'y_('+str(a)+', '+str(m)+')')
                    bqm.quadratic[key] = -2*SupplierQubo.lagrange_a*m*self.indicators[i-1][a-1]
        
        return bqm

    def _post_process(self, solutions):
        """Hack to return data formatted as expected"""
        res = []
        for solution in solutions:
            res.append([solution[i] for i in self.x])
        return res


if __name__ == "__main__":

    # from dimod import ExactSolver
    from dwave.system import LeapHybridSampler
    from neal import SimulatedAnnealingSampler
    from utils.data import read_inventory_optimization_data
    from config import standard_mock_data
    
    inventory, supplier_inventories = read_inventory_optimization_data(standard_mock_data['small'])
    # sampler = SimulatedAnnealingSampler().sample
    sampler = LeapHybridSampler().sample

    qubo = SupplierQubo(inventory, supplier_inventories)
    # qubo.solve(sampler, **{"num_reads":100, "num_sweeps": 100})
    qubo.solve(sampler)

    # print(qubo.response)

    print('\nLength of solution', len(qubo.solution_set[0]))
    print(f'Found {np.sum(qubo.solution_set[0])} suppliers with energy {qubo.energy_set[0]}')
    print('\nAll solutions: ', qubo.solution_set)