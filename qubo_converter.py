"""
QUBO to Ising Model Converter
Converts portfolio optimization problem to quantum-compatible format
Following the research.md methodology and Notes.md converter analysis
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QUBOIsingConverter:
    """
    Converts portfolio optimization problem to QUBO and Ising formats
    Following the exact methodology from research.md
    """
    
    def __init__(self):
        self.qubo_matrices = None
        self.ising_matrices = None
        self.pauli_op = None
    
    def problem_to_qubo(self, problem_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert classical problem to QUBO format
        Following research.md: H(x) = x^T Q x + h^T x + constant
        """
        Q = problem_matrices['Q']
        h = problem_matrices['h']
        constant = problem_matrices['constant']
        n = problem_matrices['n_variables']
        
        logger.info(f"Converting problem to QUBO format with {n} variables")
        
        # QUBO format: minimize x^T Q_qubo x + h_qubo^T x + constant
        # where x_i ∈ {0, 1}
        
        self.qubo_matrices = {
            'Q': Q.copy(),
            'h': h.copy(),
            'constant': constant,
            'n_variables': n,
            'offset': constant
        }
        
        logger.info(f"QUBO conversion complete")
        return self.qubo_matrices
    
    def qubo_to_ising(self, qubo_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert QUBO to Ising format using substitution x = (1 - z)/2
        Following research.md: H(z) = Σ h_i Z_i + Σ J_ij Z_i Z_j
        where Z_i ∈ {-1, +1}
        """
        Q = qubo_matrices['Q']
        h_qubo = qubo_matrices['h']
        constant = qubo_matrices['constant']
        n = qubo_matrices['n_variables']
        
        logger.info(f"Converting QUBO to Ising format")
        
        # Substitution: x_i = (1 - z_i)/2, where z_i ∈ {-1, +1}
        # x_i x_j = (1 - z_i)(1 - z_j)/4 = (1 - z_i - z_j + z_i z_j)/4
        
        # Initialize Ising parameters
        h_ising = np.zeros(n)
        J_ising = np.zeros((n, n))
        constant_ising = constant
        
        # Convert quadratic terms
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal terms: Q_ii * x_i^2 = Q_ii * (1 - z_i)^2/4
                    # = Q_ii * (1 - 2*z_i + z_i^2)/4 = Q_ii * (1 - 2*z_i + 1)/4
                    # = Q_ii * (2 - 2*z_i)/4 = Q_ii * (1 - z_i)/2
                    h_ising[i] += -Q[i, i] / 2
                    constant_ising += Q[i, i] / 2
                else:
                    # Off-diagonal terms: Q_ij * x_i * x_j 
                    # = Q_ij * (1 - z_i - z_j + z_i z_j)/4
                    J_ising[i, j] += Q[i, j] / 4
                    h_ising[i] += -Q[i, j] / 4
                    h_ising[j] += -Q[i, j] / 4
                    constant_ising += Q[i, j] / 4
        
        # Convert linear terms
        for i in range(n):
            # h_i * x_i = h_i * (1 - z_i)/2 = h_i/2 - h_i*z_i/2
            h_ising[i] += -h_qubo[i] / 2
            constant_ising += h_qubo[i] / 2
        
        # Make J matrix symmetric and set diagonal to zero
        for i in range(n):
            J_ising[i, i] = 0  # Ising J matrix has zero diagonal
            for j in range(i+1, n):
                J_sym = (J_ising[i, j] + J_ising[j, i]) / 2
                J_ising[i, j] = J_sym
                J_ising[j, i] = J_sym
        
        self.ising_matrices = {
            'h': h_ising,
            'J': J_ising,
            'constant': constant_ising,
            'n_variables': n,
            'offset': constant_ising
        }
        
        logger.info(f"Ising conversion complete")
        logger.info(f"Ising field strengths (h): min={np.min(h_ising):.4f}, max={np.max(h_ising):.4f}")
        logger.info(f"Ising couplings (J): min={np.min(J_ising):.4f}, max={np.max(J_ising):.4f}")
        
        return self.ising_matrices
    
    def ising_to_pauli(self, ising_matrices: Dict[str, Any]) -> SparsePauliOp:
        """
        Convert Ising model to Pauli operator format for Qiskit
        H = Σ h_i Z_i + Σ J_ij Z_i Z_j + constant
        """
        h = ising_matrices['h']
        J = ising_matrices['J']
        constant = ising_matrices['constant']
        n = ising_matrices['n_variables']
        
        logger.info(f"Converting Ising to Pauli operators")
        
        # Build list of Pauli terms
        pauli_list = []
        coeffs = []
        
        # Add constant term (identity)
        if abs(constant) > 1e-10:
            pauli_list.append('I' * n)
            coeffs.append(constant)
        
        # Add linear terms (Z_i)
        for i in range(n):
            if abs(h[i]) > 1e-10:
                pauli_str = 'I' * n
                pauli_str = pauli_str[:i] + 'Z' + pauli_str[i+1:]
                pauli_list.append(pauli_str)
                coeffs.append(h[i])
        
        # Add quadratic terms (Z_i Z_j)
        for i in range(n):
            for j in range(i+1, n):
                if abs(J[i, j]) > 1e-10:
                    pauli_str = 'I' * n
                    pauli_str = pauli_str[:i] + 'Z' + pauli_str[i+1:]
                    pauli_str = pauli_str[:j] + 'Z' + pauli_str[j+1:]
                    pauli_list.append(pauli_str)
                    coeffs.append(J[i, j])
        
        # Create SparsePauliOp
        if pauli_list:
            self.pauli_op = SparsePauliOp(pauli_list, coeffs=coeffs)
        else:
            # Empty operator
            self.pauli_op = SparsePauliOp('I' * n, coeffs=[0.0])
        
        logger.info(f"Created Pauli operator with {len(pauli_list)} terms")
        return self.pauli_op
    
    def convert_solution_back(self, ising_solution: np.ndarray) -> np.ndarray:
        """
        Convert Ising solution back to binary solution
        z_i ∈ {-1, +1} -> x_i = (1 - z_i)/2 ∈ {0, 1}
        """
        binary_solution = (1 - ising_solution) / 2
        return binary_solution.astype(int)
    
    def evaluate_ising_energy(self, z: np.ndarray, ising_matrices: Dict[str, Any]) -> float:
        """
        Evaluate Ising Hamiltonian energy for a given configuration
        E = Σ h_i z_i + Σ J_ij z_i z_j + constant
        """
        h = ising_matrices['h']
        J = ising_matrices['J']
        constant = ising_matrices['constant']
        
        energy = constant
        energy += np.dot(h, z)
        energy += np.dot(z, np.dot(J, z))
        
        return energy
    
    def evaluate_qubo_energy(self, x: np.ndarray, qubo_matrices: Dict[str, Any]) -> float:
        """
        Evaluate QUBO objective for a given binary configuration
        E = x^T Q x + h^T x + constant
        """
        Q = qubo_matrices['Q']
        h = qubo_matrices['h']
        constant = qubo_matrices['constant']
        
        energy = constant
        energy += np.dot(h, x)
        energy += np.dot(x, np.dot(Q, x))
        
        return energy
    
    def full_conversion_pipeline(self, problem_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete conversion pipeline from problem to all quantum formats
        """
        logger.info("Starting full conversion pipeline")
        
        # Step 1: Convert to QUBO
        qubo_matrices = self.problem_to_qubo(problem_matrices)
        
        # Step 2: Convert QUBO to Ising
        ising_matrices = self.qubo_to_ising(qubo_matrices)
        
        # Step 3: Convert to Pauli operators
        pauli_op = self.ising_to_pauli(ising_matrices)
        
        return {
            'qubo': qubo_matrices,          # full QUBO info dict
            'ising': ising_matrices,        # full Ising info dict
            'pauli_op': pauli_op,           # Pauli operator for Qiskit

            # --- Back-compatibility aliases for legacy notebooks ---
            'qubo_matrix': qubo_matrices['Q'],          # original code expected raw Q matrix
            'num_qubits': qubo_matrices['n_variables'], # number of qubits required
            'hamiltonian': pauli_op,                    # alias of pauli_op
            # ------------------------------------------------------

            'converter': self
        }
    
    def convert_full_pipeline(self, problem_matrices: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """
        Backwards-compatibility wrapper for legacy notebooks that call
        `convert_full_pipeline(problem_matrices, portfolio_vars)`.
        Any additional positional/keyword arguments beyond the first are
        ignored as the conversion relies solely on `problem_matrices`.
        """
        logger.debug("convert_full_pipeline called – redirecting to full_conversion_pipeline")
        return self.full_conversion_pipeline(problem_matrices)

    def save_matrices(self, output_dir: str = "."):
        """
        Save conversion matrices as numpy files
        Following the h.npy, J.npy format mentioned in problem.md
        """
        import os
        
        if self.ising_matrices:
            h_path = os.path.join(output_dir, "h.npy")
            J_path = os.path.join(output_dir, "J.npy")
            
            np.save(h_path, self.ising_matrices['h'])
            np.save(J_path, self.ising_matrices['J'])
            
            logger.info(f"Saved Ising matrices: {h_path}, {J_path}")
            
            # Save additional info
            info = {
                'constant': self.ising_matrices['constant'],
                'n_variables': self.ising_matrices['n_variables']
            }
            
            info_path = os.path.join(output_dir, "ising_info.npy")
            np.save(info_path, info)
            logger.info(f"Saved Ising info: {info_path}")

def main():
    """
    Test the QUBO/Ising converter
    """
    from data_processor import PortfolioDataProcessor
    
    # Load data
    processor = PortfolioDataProcessor()
    processor.load_data()
    processor.extract_portfolio_variables()
    problem_matrices = processor.create_problem_matrices()
    
    # Convert to quantum formats
    converter = QUBOIsingConverter()
    conversion_results = converter.full_conversion_pipeline(problem_matrices)
    
    # Test solution conversion
    n = problem_matrices['n_variables']
    test_ising = np.random.choice([-1, 1], n)
    test_binary = converter.convert_solution_back(test_ising)
    
    print(f"\nConversion Results:")
    print(f"Original problem variables: {n}")
    print(f"QUBO Q matrix shape: {conversion_results['qubo']['Q'].shape}")
    print(f"Ising h vector shape: {conversion_results['ising']['h'].shape}")
    print(f"Ising J matrix shape: {conversion_results['ising']['J'].shape}")
    pauli_op = conversion_results['pauli_op']
    num_terms = len(pauli_op) if hasattr(pauli_op, '__len__') else len(pauli_op.paulis) if hasattr(pauli_op, 'paulis') else 'unknown'
    print(f"Pauli operator terms: {num_terms}")
    print(f"Test conversion: Ising {test_ising[:5]}... -> Binary {test_binary[:5]}...")
    
    # Save matrices
    converter.save_matrices()
    
    return converter, conversion_results

if __name__ == "__main__":
    main()
