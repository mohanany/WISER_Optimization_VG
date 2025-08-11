"""
Variational Quantum Eigensolver (VQE) Implementation
Portfolio optimization using VQE following research.md methodology
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
from qiskit.primitives import Estimator
# Primitives will be imported conditionally based on backend choice
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# IBM Quantum
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator
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

class VQEPortfolioSolver:
    """
    VQE-based portfolio optimization solver
    Implementing best practices from research.md
    """
    
    def __init__(self, backend_type: str = "simulator", 
                 use_noise: bool = False,
                 shots: int = 1024):
        """
        Initialize VQE solver
        
        Args:
            backend_type: 'simulator', 'ibm_hardware', or 'ibm_simulator'
            use_noise: Whether to add noise model to simulation
            shots: Number of shots for quantum execution
        """
        self.backend_type = backend_type
        self.use_noise = use_noise
        self.shots = shots
        self.backend = None
        self.estimator = None
        self.results = {}
        self.optimization_history = []
        
        # VQE parameters following research.md recommendations
        self.ansatz_type = "RealAmplitudes"  # RY ansatz preferred in research.md
        self.optimizer_type = "COBYLA"      # Best performance in research.md
        self.reps = 3                       # p=3-4 layers recommended
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
                
                from qiskit_aer.primitives import Estimator as AerEstimator
                # Remove backend parameter as it's not supported in current Aer version
                self.estimator = AerEstimator()
                logger.info(f"Using AerSimulator with noise: {self.use_noise}")
                
            elif self.backend_type == "ibm_hardware" or self.backend_type == "ibm_simulator":
                if not IBM_AVAILABLE:
                    logger.warning("IBM Quantum not available, falling back to simulator")
                    self.backend = AerSimulator()
                    from qiskit_aer.primitives import Estimator as AerEstimator
                    # Remove backend parameter as it's not supported in current Aer version
                    self.estimator = AerEstimator()
                    return
                
                # Load IBM credentials
                token = os.getenv('IBM_API_TOKEN')
                crn = os.getenv('IBM_CRN')
                instance = os.getenv('IBM_INSTANCE', 'ibm-q-network/deployed/main')
                
                if not token:
                    logger.warning("IBM token not found, using simulator")
                    self.backend = AerSimulator()
                    self.estimator = Estimator()
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
                logger.info(f"Using IBM backend: {self.backend.name}")
                
        except Exception as e:
            logger.error(f"Error setting up backend: {e}")
            # Fallback to simulator
            self.backend = AerSimulator()
            self.estimator = Estimator()
            logger.info("Falling back to AerSimulator")
    
    def create_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """
        Create quantum ansatz circuit
        Following research.md recommendations: RY ansatz with p=3-4 layers
        """
        if self.ansatz_type == "RealAmplitudes":
            # RY ansatz - preferred in research.md
            ansatz = RealAmplitudes(num_qubits, reps=self.reps, entanglement='linear')
            
        elif self.ansatz_type == "TwoLocal":
            # Alternative: PauliTwo ansatz mentioned in research.md
            ansatz = TwoLocal(num_qubits, 
                             rotation_blocks=['ry', 'rz'], 
                             entanglement_blocks='cz',
                             entanglement='linear', 
                             reps=self.reps)
        else:
            # Default to RealAmplitudes
            ansatz = RealAmplitudes(num_qubits, reps=self.reps, entanglement='linear')
        
        logger.info(f"Created {self.ansatz_type} ansatz with {ansatz.num_parameters} parameters")
        return ansatz
    
    def create_optimizer(self) -> Any:
        """
        Create classical optimizer
        Following research.md: COBYLA preferred, lr=0.05-0.1 for Adam
        """
        if self.optimizer_type == "COBYLA":
            optimizer = COBYLA(maxiter=self.maxiter)
            
        elif self.optimizer_type == "Adam":
            optimizer = Adam(maxiter=self.maxiter, lr=0.1)  # research.md recommendation
            
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
            logger.info(f"VQE Iteration {iteration}: Objective = {objective_value:.6f}")
    
    def solve_vqe(self, conversion_results: Dict[str, Any], 
                  warm_start_params: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Solve portfolio optimization using VQE
        
        Args:
            conversion_results: Results from QUBO/Ising conversion
            warm_start_params: Optional initial parameters for warm start
            
        Returns:
            Dictionary with VQE solution results
        """
        start_time = time.time()
        
        try:
            # Get Pauli operator
            pauli_op = conversion_results['pauli_op']
            num_qubits = conversion_results['ising']['n_variables']
            
            logger.info(f"Starting VQE optimization for {num_qubits} qubits")
            
            # Create ansatz and optimizer
            ansatz = self.create_ansatz(num_qubits)
            optimizer = self.create_optimizer()
            
            # Initial parameters
            if warm_start_params is not None and len(warm_start_params) == ansatz.num_parameters:
                initial_point = warm_start_params
                logger.info("Using warm start parameters")
            else:
                # Random initialization with small values
                initial_point = np.random.uniform(-np.pi/4, np.pi/4, ansatz.num_parameters)
                logger.info("Using random initial parameters")
            
            # Create VQE instance
            vqe = VQE(
                estimator=self.estimator,
                ansatz=ansatz,
                optimizer=optimizer,
                initial_point=initial_point,
                callback=self.objective_callback
            )
            
            # Clear optimization history
            self.optimization_history = []
            
            # Run VQE
            logger.info("Running VQE optimization...")
            vqe_result = vqe.compute_minimum_eigenvalue(pauli_op)
            
            solve_time = time.time() - start_time
            
            # Extract solution
            optimal_parameters = vqe_result.optimal_parameters
            optimal_value = vqe_result.optimal_value
            
            # Get quantum state and measurements
            optimal_circuit = ansatz.assign_parameters(optimal_parameters)
            
            # Sample from optimal state to get binary solution
            binary_solution = self._sample_solution(optimal_circuit, num_qubits)
            
            # Convert back to original problem format
            converter = conversion_results['converter']
            original_solution = converter.convert_solution_back(
                2 * binary_solution - 1  # Convert {0,1} to {-1,+1}
            )
            
            # Calculate solution quality
            qubo_energy = converter.evaluate_qubo_energy(
                original_solution, conversion_results['qubo']
            )
            
            # Prepare results
            results = {
                'success': True,
                'optimal_value': optimal_value,
                'optimal_parameters': optimal_parameters,
                'binary_solution': binary_solution,
                'original_solution': original_solution,
                'qubo_energy': qubo_energy,
                'solve_time': solve_time,
                'num_iterations': len(self.optimization_history),
                'solver': 'VQE',
                'backend': self.backend.name if hasattr(self.backend, 'name') else str(self.backend),
                'ansatz_type': self.ansatz_type,
                'optimizer_type': self.optimizer_type,
                'optimization_history': self.optimization_history.copy(),
                'selected_assets': np.where(original_solution > 0.5)[0],
                'num_selected': np.sum(original_solution > 0.5)
            }
            
            self.results = results
            
            logger.info(f"VQE completed in {solve_time:.2f}s")
            logger.info(f"Optimal value: {optimal_value:.6f}")
            logger.info(f"Selected {results['num_selected']} assets")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in VQE optimization: {e}")
            solve_time = time.time() - start_time
            
            return {
                'success': False,
                'error': str(e),
                'solve_time': solve_time,
                'solver': 'VQE',
                'backend': self.backend.name if hasattr(self.backend, 'name') else str(self.backend)
            }
    
    def _sample_solution(self, circuit: QuantumCircuit, num_qubits: int, 
                        num_samples: int = 1000) -> np.ndarray:
        """
        Sample binary solution from quantum state
        """
        try:
            # Add measurements to circuit
            circuit_with_measurements = circuit.copy()
            circuit_with_measurements.add_register(circuit_with_measurements.cregs[0] 
                                                  if circuit_with_measurements.cregs 
                                                  else circuit_with_measurements.add_register('c', num_qubits))
            circuit_with_measurements.measure_all()
            
            # Run circuit
            if hasattr(self.backend, 'run'):
                job = self.backend.run(circuit_with_measurements, shots=num_samples)
                result = job.result()
                counts = result.get_counts()
            else:
                # For IBM Runtime backends
                from qiskit_ibm_runtime import Sampler
                sampler = Sampler(backend=self.backend)
                job = sampler.run(circuit_with_measurements, shots=num_samples)
                result = job.result()
                counts = result.quasi_dists[0].binary_probabilities()
            
            # Find most probable bitstring
            max_count = 0
            best_bitstring = '0' * num_qubits
            
            for bitstring, count in counts.items():
                if isinstance(count, float):
                    count = int(count * num_samples)
                if count > max_count:
                    max_count = count
                    best_bitstring = bitstring
            
            # Convert to numpy array
            binary_solution = np.array([int(b) for b in best_bitstring[::-1]])  # Reverse for qubit ordering
            
            return binary_solution
            
        except Exception as e:
            logger.warning(f"Error sampling solution, using random: {e}")
            # Return random solution as fallback
            return np.random.randint(0, 2, num_qubits)
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot VQE optimization convergence
        """
        if not self.optimization_history:
            logger.warning("No optimization history to plot")
            return
        
        iterations = [entry['iteration'] for entry in self.optimization_history]
        objectives = [entry['objective_value'] for entry in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, objectives, 'b-', marker='o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(f'VQE Optimization Convergence ({self.backend_type})')
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
    Test VQE solver
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
    
    # Test VQE with simulator
    vqe_solver = VQEPortfolioSolver(backend_type="simulator", use_noise=False, shots=1024)
    vqe_results = vqe_solver.solve_vqe(conversion_results)
    
    # Print results
    print(f"\nVQE Results:")
    print(f"Success: {vqe_results['success']}")
    if vqe_results['success']:
        print(f"Backend: {vqe_results['backend']}")
        print(f"Solve time: {vqe_results['solve_time']:.2f}s")
        print(f"Iterations: {vqe_results['num_iterations']}")
        print(f"Optimal value: {vqe_results['optimal_value']:.6f}")
        print(f"QUBO energy: {vqe_results['qubo_energy']:.6f}")
        print(f"Selected assets: {vqe_results['num_selected']}")
        print(f"Asset indices: {vqe_results['selected_assets']}")
        
        # Plot convergence
        vqe_solver.plot_optimization_history()
    else:
        print(f"Error: {vqe_results['error']}")
    
    return vqe_solver, vqe_results

if __name__ == "__main__":
    main()
