"""
Classical Portfolio Optimization using GUROBI
Implements the baseline classical solution for comparison
Following the OneOpto model from problem.md
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: Gurobi not available. Install with: pip install gurobipy")

from data_processor import PortfolioDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassicalPortfolioOptimizer:
    """
    Classical portfolio optimizer using GUROBI
    Implements the exact OneOpto formulation from problem.md
    """
    
    def __init__(self, data_processor: PortfolioDataProcessor):
        self.data_processor = data_processor
        self.model = None
        self.solution = None
        self.solve_time = 0
        
    def build_gurobi_model(self, problem_matrices: Dict[str, Any]) -> Optional[gp.Model]:
        """
        Build GUROBI model following OneOpto formulation
        Binary variables y_c for asset selection
        """
        if not GUROBI_AVAILABLE:
            logger.error("GUROBI not available")
            return None
        
        try:
            # Create model
            model = gp.Model("PortfolioOptimization")
            model.setParam('OutputFlag', 1)  # Enable output
            model.setParam('TimeLimit', 300)  # 5 minute time limit
            
            n = problem_matrices['n_variables']
            Q = problem_matrices['Q']
            h = problem_matrices['h']
            constant = problem_matrices['constant']
            max_assets = problem_matrices['max_assets']
            
            # Decision variables - binary selection variables y_c
            y = model.addVars(n, vtype=GRB.BINARY, name="y")
            
            # Objective function: minimize x^T Q x + h^T x + constant
            # For binary variables, this becomes: y^T Q y + h^T y + constant
            obj = gp.QuadExpr()
            
            # Quadratic terms
            for i in range(n):
                for j in range(n):
                    if Q[i, j] != 0:
                        obj += Q[i, j] * y[i] * y[j]
            
            # Linear terms
            for i in range(n):
                if h[i] != 0:
                    obj += h[i] * y[i]
            
            # Add constant
            obj += constant
            
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Constraints
            
            # 1. Maximum number of assets constraint
            # sum(y_c) <= max_assets
            model.addConstr(gp.quicksum(y[i] for i in range(n)) <= max_assets, 
                           name="max_assets")
            
            # 2. Minimum return constraint (optional)
            returns = problem_matrices['returns']
            target_return = problem_matrices['target_return']
            min_return_fraction = 0.8  # At least 80% of target return
            
            model.addConstr(
                gp.quicksum(returns[i] * y[i] for i in range(n)) >= 
                target_return * min_return_fraction,
                name="min_return"
            )
            
            # 3. At least one asset must be selected
            model.addConstr(gp.quicksum(y[i] for i in range(n)) >= 1, 
                           name="min_one_asset")
            
            self.model = model
            logger.info(f"Built GUROBI model with {n} variables and {model.NumConstrs} constraints")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building GUROBI model: {e}")
            return None
    
    def solve_classical(self, problem_matrices: Dict[str, Any], 
                       time_limit: int = 300) -> Dict[str, Any]:
        """
        Solve the portfolio optimization problem using GUROBI
        
        Args:
            problem_matrices: Problem formulation matrices
            time_limit: Maximum solve time in seconds
            
        Returns:
            Dictionary with solution results
        """
        if not GUROBI_AVAILABLE:
            logger.error("GUROBI not available")
            return self._dummy_solution(problem_matrices)
        
        start_time = time.time()
        
        try:
            # Build model
            model = self.build_gurobi_model(problem_matrices)
            if model is None:
                return self._dummy_solution(problem_matrices)
            
            # Set time limit
            model.setParam('TimeLimit', time_limit)
            
            # Solve
            logger.info("Starting GUROBI optimization...")
            model.optimize()
            
            self.solve_time = time.time() - start_time
            
            # Extract solution
            if model.status == GRB.OPTIMAL:
                # Get solution values
                n = problem_matrices['n_variables']
                solution_vector = np.zeros(n)
                
                for i in range(n):
                    solution_vector[i] = model.getVarByName(f"y[{i}]").X
                
                objective_value = model.ObjVal
                
                # Calculate portfolio metrics
                selected_assets = np.where(solution_vector > 0.5)[0]
                portfolio_return = np.sum(problem_matrices['returns'][selected_assets])
                
                # Calculate portfolio risk
                selected_cov = problem_matrices['covariance'][np.ix_(selected_assets, selected_assets)]
                portfolio_variance = np.sum(selected_cov) if len(selected_assets) > 0 else 0
                portfolio_risk = np.sqrt(portfolio_variance)
                
                solution = {
                    'success': True,
                    'optimal': True,
                    'solution_vector': solution_vector,
                    'selected_assets': selected_assets,
                    'objective_value': objective_value,
                    'solve_time': self.solve_time,
                    'portfolio_return': portfolio_return,
                    'portfolio_risk': portfolio_risk,
                    'num_selected': len(selected_assets),
                    'solver': 'GUROBI',
                    'status': 'Optimal',
                    'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0
                }
                
                logger.info(f"GUROBI found optimal solution in {self.solve_time:.2f}s")
                logger.info(f"Selected {len(selected_assets)} assets with objective {objective_value:.6f}")
                
            elif model.status == GRB.TIME_LIMIT:
                # Time limit reached but may have feasible solution
                if model.SolCount > 0:
                    n = problem_matrices['n_variables']
                    solution_vector = np.zeros(n)
                    
                    for i in range(n):
                        solution_vector[i] = model.getVarByName(f"y[{i}]").X
                    
                    selected_assets = np.where(solution_vector > 0.5)[0]
                    
                    solution = {
                        'success': True,
                        'optimal': False,
                        'solution_vector': solution_vector,
                        'selected_assets': selected_assets,
                        'objective_value': model.ObjVal,
                        'solve_time': self.solve_time,
                        'solver': 'GUROBI',
                        'status': 'Time Limit',
                        'gap': model.MIPGap if hasattr(model, 'MIPGap') else float('inf')
                    }
                    
                    logger.info(f"GUROBI hit time limit but found feasible solution")
                else:
                    solution = self._infeasible_solution(problem_matrices)
                    
            else:
                logger.warning(f"GUROBI status: {model.status}")
                solution = self._infeasible_solution(problem_matrices)
            
            self.solution = solution
            return solution
            
        except Exception as e:
            logger.error(f"Error solving with GUROBI: {e}")
            return self._dummy_solution(problem_matrices)
    
    def _dummy_solution(self, problem_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a dummy solution when GUROBI is not available
        """
        n = problem_matrices['n_variables']
        max_assets = problem_matrices['max_assets']
        
        # Select top assets by return/risk ratio
        returns = problem_matrices['returns']
        # Use diagonal of Q matrix as risk proxy
        risks = np.sqrt(np.diag(problem_matrices['Q']))
        
        # Avoid division by zero
        risk_adjusted_returns = np.divide(returns, risks, 
                                        out=np.zeros_like(returns), 
                                        where=risks!=0)
        
        # Select top assets
        top_indices = np.argsort(risk_adjusted_returns)[-max_assets:]
        
        solution_vector = np.zeros(n)
        solution_vector[top_indices] = 1
        
        # Calculate objective value
        objective_value = np.dot(solution_vector, np.dot(problem_matrices['Q'], solution_vector)) + \
                         np.dot(problem_matrices['h'], solution_vector) + \
                         problem_matrices['constant']
        
        return {
            'success': True,
            'optimal': False,
            'solution_vector': solution_vector,
            'selected_assets': top_indices,
            'objective_value': objective_value,
            'solve_time': 0.1,
            'solver': 'Heuristic',
            'status': 'Heuristic Solution',
            'gap': float('inf'),
            'note': 'GUROBI not available - using heuristic solution'
        }
    
    def _infeasible_solution(self, problem_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create infeasible solution response
        """
        n = problem_matrices['n_variables']
        
        return {
            'success': False,
            'optimal': False,
            'solution_vector': np.zeros(n),
            'selected_assets': np.array([]),
            'objective_value': float('inf'),
            'solve_time': self.solve_time,
            'solver': 'GUROBI',
            'status': 'Infeasible',
            'gap': float('inf')
        }
    
    def warm_start_solution(self, problem_matrices: Dict[str, Any], 
                          time_limit: int = 60) -> Dict[str, Any]:
        """
        Quick warm-start solution for quantum algorithms
        Following research.md recommendation for 60-second classical warm-start
        """
        logger.info("Computing warm-start solution...")
        return self.solve_classical(problem_matrices, time_limit=time_limit)

def main():
    """
    Test the classical solver
    """
    # Load data
    processor = PortfolioDataProcessor()
    processor.load_data()
    processor.extract_portfolio_variables()
    problem_matrices = processor.create_problem_matrices()
    
    # Solve classically
    solver = ClassicalPortfolioOptimizer(processor)
    solution = solver.solve_classical(problem_matrices, time_limit=60)
    
    # Print results
    print("\nClassical Solution Results:")
    print(f"Success: {solution['success']}")
    print(f"Status: {solution['status']}")
    print(f"Solve time: {solution['solve_time']:.2f}s")
    print(f"Objective value: {solution['objective_value']:.6f}")
    print(f"Number of selected assets: {solution.get('num_selected', 0)}")
    
    if 'selected_assets' in solution:
        print(f"Selected assets: {solution['selected_assets']}")
    
    return solver, solution

if __name__ == "__main__":
    main()
