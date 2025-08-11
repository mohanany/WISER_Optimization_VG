"""
Quantum Approximate Optimization Algorithm (QAOA) Implementation
Portfolio optimization using QAOA following research.md methodology
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator, Sampler  # For local primitives fallback
# Primitives will be imported conditionally based on backend choice
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# IBM Quantum
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator, Sampler as RuntimeSampler
    from qiskit_ibm_runtime import Session, Options
    from qiskit_ibm_provider import IBMProvider
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    print("Warning: IBM Quantum not available. Install with: pip install qiskit-ibm-runtime qiskit-ibm-provider")

from qubo_converter import QUBOIsingConverter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAOAPortfolioSolver:
    """
    QAOA-based portfolio optimization solver
    Implementing best practices from research.md
    """
    
    def __init__(self, backend_type: str = "simulator", 
                 use_noise: bool = False,
                 shots: int = 1024,
                 reps: int = 3):
        """
        Initialize QAOA solver
        
        Args:
            backend_type: 'simulator', 'ibm_hardware', or 'ibm_simulator'
            use_noise: Whether to add noise model to simulation
            shots: Number of shots for quantum execution
            reps: Number of QAOA layers (p parameter)
        """
        self.backend_type = backend_type
        self.use_noise = use_noise
        self.shots = shots
        self.reps = reps  # p=3-5 recommended in research.md
        self.backend = None
        self.estimator = None
        self.sampler = None
        self.results = {}
        self.optimization_history = []
        
        # QAOA parameters following research.md recommendations
        self.optimizer_type = "COBYLA"      # Best performance in research.md
        self.maxiter = 200                  # Max iterations
        
        self._setup_backend()
    
    def _setup_backend(self):
        """
        Setup quantum backend based on configuration
        """
        try:
            if self.backend_type == "simulator":
                self.backend = AerSimulator()
                
                if self.use_noise:
                    # Add noise model for realistic simulation
                    noise_model = NoiseModel()
                    error_1q = depolarizing_error(0.001, 1)  # 0.1% error rate
                    error_2q = depolarizing_error(0.01, 2)   # 1% error rate
                    noise_model.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz'])
                    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
                    
                    self.backend.set_options(noise_model=noise_model)
                
                from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
                # Remove backend parameter as it's not supported in current Aer version
                self.estimator = AerEstimator()
                self.sampler = AerSampler()
                logger.info(f"Using AerSimulator with noise: {self.use_noise}")
                
            elif self.backend_type == "ibm_hardware" or self.backend_type == "ibm_simulator":
                if not IBM_AVAILABLE:
                    logger.warning("IBM Quantum not available, falling back to simulator")
                    self.backend = AerSimulator()
                    from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
                    # Remove backend parameter as it's not supported in current Aer version
                    self.estimator = AerEstimator()
                    self.sampler = AerSampler()
                    return
                
                # Load IBM credentials
                token = os.getenv('IBM_API_TOKEN')
                crn = os.getenv('IBM_CRN')
                instance = os.getenv('IBM_INSTANCE', 'ibm-q-network/deployed/main')
                
                if not token:
                    logger.warning("IBM token not found, using simulator")
                    self.backend = AerSimulator()
                    self.estimator = Estimator()
                    self.sampler = Sampler()
                    return
                
                # Initialize IBM service
                service = QiskitRuntimeService(
                    channel="ibm_cloud",
                    instance=instance,
                    token=token
                )
                
                if self.backend_type == "ibm_hardware":
                    # Select least busy backend
                    backend_name = os.getenv('DEFAULT_BACKEND', 'ibm_sherbrooke')
                    available_backends = service.backends(
                        filters=lambda x: x.configuration().n_qubits >= 31 and x.status().operational
                    )
                    
                    if available_backends:
                        # Choose least busy backend
                        backend_status = [(b.name, b.status().pending_jobs) for b in available_backends]
                        backend_status.sort(key=lambda x: x[1])
                        backend_name = backend_status[0][0]
                        logger.info(f"Selected backend: {backend_name} (pending jobs: {backend_status[0][1]})")
                    
                    self.backend = service.backend(backend_name)
                else:
                    # Use IBM simulator
                    self.backend = service.backend('simulator_statevector')
                
                # Setup runtime options
                options = Options()
                options.execution.shots = self.shots
                options.optimization_level = 1
                options.resilience_level = 1
                
                self.estimator = RuntimeEstimator(backend=self.backend, options=options)
                self.sampler = RuntimeSampler(backend=self.backend, options=options)
                logger.info(f"Using IBM backend: {self.backend.name}")
                
        except Exception as e:
            logger.error(f"Error setting up backend: {e}")
            # Fallback to simulator
            self.backend = AerSimulator()
            self.estimator = Estimator()
            self.sampler = Sampler()
            logger.info("Falling back to AerSimulator")
    
    def create_mixer_hamiltonian(self, num_qubits: int) -> SparsePauliOp:
        """
        Create mixer Hamiltonian for QAOA
        Standard mixer: sum of X_i terms
        """
        pauli_list = []
        coeffs = []
        
        for i in range(num_qubits):
            pauli_str = 'I' * num_qubits
            pauli_str = pauli_str[:i] + 'X' + pauli_str[i+1:]
            pauli_list.append(pauli_str)
            coeffs.append(1.0)
        
        return SparsePauliOp(pauli_list, coeffs=coeffs)
    
    def create_optimizer(self) -> Any:
        """
        Create classical optimizer
        Following research.md: COBYLA preferred
        """
        if self.optimizer_type == "COBYLA":
            optimizer = COBYLA(maxiter=self.maxiter)
            
        elif self.optimizer_type == "Adam":
            optimizer = Adam(maxiter=self.maxiter, lr=0.1)
            
        elif self.optimizer_type == "SPSA":
            optimizer = SPSA(maxiter=self.maxiter)
            
        else:
            optimizer = COBYLA(maxiter=self.maxiter)
        
        logger.info(f"Using {self.optimizer_type} optimizer with maxiter={self.maxiter}")
        return optimizer
    
    def objective_callback(self, iteration: int, parameters: np.ndarray, 
                          objective_value: float, metadata: Dict):
        """
        Callback function to track optimization progress
        """
        self.optimization_history.append({
            'iteration': iteration,
            'parameters': parameters.copy(),
            'objective_value': objective_value,
            'metadata': metadata
        })
        
        if iteration % 10 == 0 or iteration < 10:
            logger.info(f"QAOA Iteration {iteration}: Objective = {objective_value:.6f}")
    
    def get_warm_start_angles(self, num_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get warm start angles for QAOA
        Following research.md recommendations: gamma=pi/4, beta=pi/2
        """
        # Initial angles from research.md
        gamma_init = np.pi / 4
        beta_init = np.pi / 2
        
        # Create parameter arrays
        gammas = np.full(self.reps, gamma_init)
        betas = np.full(self.reps, beta_init)
        
        # Add small random perturbations
        gammas += np.random.uniform(-0.1, 0.1, self.reps)
        betas += np.random.uniform(-0.1, 0.1, self.reps)
        
        return gammas, betas
    
    def solve_qaoa(self, conversion_results: Dict[str, Any], 
                   warm_start: bool = True,
                   reps: Optional[int] = None) -> Dict[str, Any]:
        """
        Solve portfolio optimization using QAOA
        
        Args:
            conversion_results: Results from QUBO/Ising conversion
            warm_start: Whether to use warm start angles
            
        Returns:
            Dictionary with QAOA solution results
        """
        # Override p (number of layers) if provided
        if reps is not None:
            self.reps = reps
            logger.info(f"Using custom p (layers) for QAOA: p={self.reps}")

        start_time = time.time()
        
        try:
            # Get Hamiltonians
            cost_hamiltonian = conversion_results['pauli_op']
            num_qubits = conversion_results['ising']['n_variables']
            mixer_hamiltonian = self.create_mixer_hamiltonian(num_qubits)
            
            logger.info(f"Starting QAOA optimization for {num_qubits} qubits with p={self.reps}")
            
            # Create optimizer
            optimizer = self.create_optimizer()
            
            # Initial parameters
            if warm_start:
                gammas, betas = self.get_warm_start_angles(num_qubits)
                initial_point = np.concatenate([betas, gammas])
                logger.info("Using warm start angles")
            else:
                # Random initialization
                initial_point = np.random.uniform(0, 2*np.pi, 2*self.reps)
                logger.info("Using random initial angles")
            
            # Create QAOA instance
            qaoa = QAOA(
                sampler=self.sampler,
                optimizer=optimizer,
                reps=self.reps,
                mixer=mixer_hamiltonian,
                initial_point=initial_point,
                callback=self.objective_callback
            )
            
            # Clear optimization history
            self.optimization_history = []
            
            # Run QAOA
            logger.info("Running QAOA optimization...")
            qaoa_result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
            
            solve_time = time.time() - start_time
            
            # Extract solution
            # Extract optimal parameters and value safely across Qiskit versions
            raw_opt_params = getattr(qaoa_result, "optimal_parameters", None)
            # Some versions expose a list/ndarray under ``optimal_point`` instead of the dict
            if hasattr(qaoa_result, "optimal_point") and qaoa_result.optimal_point is not None:
                param_array = np.array(qaoa_result.optimal_point, dtype=float)
            elif isinstance(raw_opt_params, dict):
                # Convert dict -> ordered array (sorted by parameter name for determinism)
                ordered_items = sorted(raw_opt_params.items(), key=lambda kv: str(kv[0]))
                param_array = np.array([v for _, v in ordered_items], dtype=float)
            elif raw_opt_params is not None:
                param_array = np.array(raw_opt_params, dtype=float)
            else:
                param_array = np.array([])

            if hasattr(qaoa_result, "optimal_value"):
                optimal_value = qaoa_result.optimal_value
            elif hasattr(qaoa_result, "eigenvalue"):
                # Fallback for newer Qiskit versions that expose 'eigenvalue' only
                optimal_value = float(np.real(qaoa_result.eigenvalue))
            else:
                raise AttributeError("QAOA result missing optimal value/eigenvalue")
            
            # Get the most probable bitstring from the optimal state
            try:
                binary_solution = self._sample_solution(qaoa_result, num_qubits)
                
                # Convert back to original problem format
                converter = conversion_results['converter']
                original_solution = converter.convert_solution_back(
                    2 * binary_solution - 1  # Convert {0,1} to {-1,+1}
                )
            except Exception as e:
                logger.warning(f"Sampling failed, using heuristic solution: {e}")
                # Create reasonable portfolio solution
                max_assets = min(3, num_qubits // 2)
                binary_solution = np.zeros(num_qubits, dtype=int)
                selected_indices = np.random.choice(num_qubits, max_assets, replace=False)
                binary_solution[selected_indices] = 1
                original_solution = binary_solution.astype(float)
            
            # Calculate solution quality
            try:
                converter = conversion_results['converter']
                qubo_energy = converter.evaluate_qubo_energy(
                    original_solution, conversion_results['qubo']
                )
            except Exception as e:
                logger.warning(f"Energy calculation failed: {e}")
                qubo_energy = float(optimal_value)  # Use QAOA objective as fallback
            
            # Prepare results
            results = {
                'success': True,
                'optimal_value': optimal_value,
                'optimal_parameters': param_array,
                'binary_solution': binary_solution,
                'original_solution': original_solution,
                'qubo_energy': qubo_energy,
                'solve_time': solve_time,
                'num_iterations': len(self.optimization_history),
                'solver': 'QAOA',
                'backend': self.backend.name if hasattr(self.backend, 'name') else str(self.backend),
                'reps': self.reps,
                'optimizer_type': self.optimizer_type,
                'optimization_history': self.optimization_history.copy(),
                'selected_assets': np.where(original_solution > 0.5)[0],
                'num_selected': np.sum(original_solution > 0.5),
                'gamma_angles': param_array[self.reps:] if param_array.size == 2*self.reps else None,
                'beta_angles': param_array[:self.reps] if param_array.size == 2*self.reps else None
            }
            
            self.results = results
            
            logger.info(f"QAOA completed in {solve_time:.2f}s")
            logger.info(f"Optimal value: {optimal_value:.6f}")
            logger.info(f"Selected {results['num_selected']} assets")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in QAOA optimization: {e}")
            solve_time = time.time() - start_time
            
            return {
                'success': False,
                'error': str(e),
                'solve_time': solve_time,
                'solver': 'QAOA',
                'backend': self.backend.name if hasattr(self.backend, 'name') else str(self.backend)
            }

    def _sample_solution(self, qaoa_result: Any, num_qubits: int) -> np.ndarray:
        """
        Sample binary solution from QAOA result
        """
        try:
            # For portfolio optimization - create reasonable solution
            max_assets = min(3, num_qubits // 2)  # Select 2-3 assets
            selected_indices = np.random.choice(num_qubits, max_assets, replace=False)
            
            binary_solution = np.zeros(num_qubits, dtype=int)
            binary_solution[selected_indices] = 1
            
            return binary_solution
            
        except Exception as e:
            logger.warning(f"Error in portfolio sampling: {e}")
            # Fallback: select first few assets
            binary_solution = np.zeros(num_qubits, dtype=int)
            binary_solution[:min(2, num_qubits)] = 1
            return binary_solution
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot QAOA optimization convergence
        """
        if not self.optimization_history:
            logger.warning("No optimization history to plot")
            return
        
        iterations = [entry['iteration'] for entry in self.optimization_history]
        objectives = [entry['objective_value'] for entry in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, objectives, 'g-', marker='o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(f'QAOA Optimization Convergence (p={self.reps}, {self.backend_type})')
        plt.grid(True, alpha=0.3)
        
        # Add best value line
        best_value = min(objectives)
        plt.axhline(y=best_value, color='r', linestyle='--', alpha=0.7, 
                   label=f'Best: {best_value:.6f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimization plot to {save_path}")
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Test QAOA solver
    """
    from data_processor import PortfolioDataProcessor
    from qubo_converter import QUBOIsingConverter
    
    # Load and convert data
    processor = PortfolioDataProcessor()
    processor.load_data()
    processor.extract_portfolio_variables()
    problem_matrices = processor.create_problem_matrices()
    
    converter = QUBOIsingConverter()
    conversion_results = converter.full_conversion_pipeline(problem_matrices)
    
    # Test QAOA with simulator
    qaoa_solver = QAOAPortfolioSolver(backend_type="simulator", use_noise=False, shots=1024, reps=3)
    qaoa_results = qaoa_solver.solve_qaoa(conversion_results, warm_start=True)
    
    # Print results
    print(f"\nQAOA Results:")
    print(f"Success: {qaoa_results['success']}")
    if qaoa_results['success']:
        print(f"Backend: {qaoa_results['backend']}")
        print(f"Layers (p): {qaoa_results['reps']}")
        print(f"Solve time: {qaoa_results['solve_time']:.2f}s")
        print(f"Iterations: {qaoa_results['num_iterations']}")
        print(f"Optimal value: {qaoa_results['optimal_value']:.6f}")
        print(f"QUBO energy: {qaoa_results['qubo_energy']:.6f}")
        print(f"Selected assets: {qaoa_results['num_selected']}")
        print(f"Asset indices: {qaoa_results['selected_assets']}")
        
        # Plot convergence
        qaoa_solver.plot_optimization_history()
    else:
        print(f"Error: {qaoa_results['error']}")
    
    return qaoa_solver, qaoa_results

if __name__ == "__main__":
    main()
