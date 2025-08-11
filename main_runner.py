"""
Main Runner Script for WISER Portfolio Optimization Challenge
Orchestrates the complete solution pipeline as specified in research.md
"""

import numpy as np
import pandas as pd
import time
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data_processor import PortfolioDataProcessor
from classical_solver import ClassicalPortfolioOptimizer
from qubo_converter import QUBOIsingConverter
from vqe_solver import VQEPortfolioSolver
from qaoa_solver import QAOAPortfolioSolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WISERPortfolioOptimizer:
    """
    Main orchestrator for the WISER portfolio optimization challenge
    Following the complete methodology from research.md
    """
    
    def __init__(self, data_path: str = "../data/1/", output_dir: str = "results/"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_processor = None
        self.classical_solver = None
        self.converter = None
        self.vqe_solver = None
        self.qaoa_solver = None
        
        # Results storage
        self.results = {}
        self.comparison_data = {}
        
        logger.info(f"Initialized Portfolio Optimizer")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def step1_load_and_process_data(self) -> Dict[str, Any]:
        """
        Step 1: Load and process portfolio data
        Following the data processing pipeline from Notes.md
        """
        logger.info("="*60)
        logger.info("STEP 1: Loading and Processing Data")
        logger.info("="*60)
        
        # Initialize data processor
        self.data_processor = PortfolioDataProcessor(self.data_path)
        
        # Load data
        assets_data, dictionary = self.data_processor.load_data()
        logger.info(f"Loaded assets data: {assets_data.shape}")
        logger.info(f"Loaded dictionary: {dictionary.shape}")
        
        # Extract portfolio variables
        portfolio_vars = self.data_processor.extract_portfolio_variables()
        
        # Create problem matrices
        problem_matrices = self.data_processor.create_problem_matrices()
        
        # Validate data
        self.data_processor.validate_data()
        
        # Save processed data
        processed_data_path = self.output_dir / "processed_data.pkl"
        self.data_processor.save_processed_data(str(processed_data_path))
        
        self.results['step1'] = {
            'portfolio_vars': portfolio_vars,
            'problem_matrices': problem_matrices,
            'data_path': processed_data_path
        }
        
        logger.info(f"Step 1 completed successfully")
        logger.info(f"Number of assets: {portfolio_vars['n_assets']}")
        logger.info(f"Max assets to select: {portfolio_vars['max_assets']}")
        logger.info(f"Target return: {problem_matrices['target_return']:.4f}")
        
        return self.results['step1']
    
    def step2_classical_baseline(self, time_limit: int = 300) -> Dict[str, Any]:
        """
        Step 2: Solve using classical GUROBI baseline
        Following the OneOpto model from problem.md
        """
        logger.info("="*60)
        logger.info("STEP 2: Classical Warm-Start Preparation")
        logger.info("="*60)
        
        if 'step1' not in self.results:
            raise ValueError("Must complete step 1 first")
        
        problem_matrices = self.results['step1']['problem_matrices']
        
        # Initialize classical solver
        self.classical_solver = ClassicalPortfolioOptimizer(self.data_processor)
        
        # Get warm start solution (60s as per research.md)
        # This provides initial guidance for quantum algorithms
        warm_start_result = self.classical_solver.warm_start_solution(
            problem_matrices, time_limit=60
        )
        
        self.results['step2'] = {
            'warm_start_solution': warm_start_result,
            'note': 'Classical used only as warm-start for quantum algorithms'
        }
        
        logger.info(f"Step 2 completed successfully")
        logger.info(f"Warm-start solution status: {warm_start_result['status']}")
        logger.info(f"Warm-start solve time: {warm_start_result['solve_time']:.2f}s")
        logger.info(f"Warm-start objective: {warm_start_result['objective_value']:.6f}")
        
        return self.results['step2']
    
    def step3_quantum_conversion(self) -> Dict[str, Any]:
        """
        Step 3: Convert problem to quantum format (QUBO/Ising)
        Following the conversion methodology from research.md
        """
        logger.info("="*60)
        logger.info("STEP 3: Quantum Problem Conversion")
        logger.info("="*60)
        
        if 'step1' not in self.results:
            raise ValueError("Must complete step 1 first")
        
        problem_matrices = self.results['step1']['problem_matrices']
        
        # Initialize converter
        self.converter = QUBOIsingConverter()
        
        # Full conversion pipeline
        conversion_results = self.converter.full_conversion_pipeline(problem_matrices)
        
        # Save matrices to files
        self.converter.save_matrices(str(self.output_dir))
        
        self.results['step3'] = conversion_results
        
        logger.info(f"Step 3 completed successfully")
        logger.info(f"QUBO matrix shape: {conversion_results['qubo']['Q'].shape}")
        logger.info(f"Ising field range: [{np.min(conversion_results['ising']['h']):.4f}, {np.max(conversion_results['ising']['h']):.4f}]")
        logger.info(f"Ising coupling range: [{np.min(conversion_results['ising']['J']):.4f}, {np.max(conversion_results['ising']['J']):.4f}]")
        pauli_op = conversion_results['pauli_op']
        num_terms = len(pauli_op) if hasattr(pauli_op, '__len__') else len(pauli_op.paulis) if hasattr(pauli_op, 'paulis') else 'unknown'
        logger.info(f"Pauli operator terms: {num_terms}")
        
        return self.results['step3']
    
    def step4_vqe_optimization(self, backend_type: str = "simulator", 
                              use_noise: bool = False, shots: int = 1024) -> Dict[str, Any]:
        """
        Step 4: VQE quantum optimization
        Following VQE methodology from research.md
        """
        logger.info("="*60)
        logger.info("STEP 4: VQE Quantum Optimization")
        logger.info("="*60)
        
        if 'step3' not in self.results:
            raise ValueError("Must complete step 3 first")
        
        conversion_results = self.results['step3']
        
        # Initialize VQE solver
        self.vqe_solver = VQEPortfolioSolver(
            backend_type=backend_type,
            use_noise=use_noise,
            shots=shots
        )
        
        # Run VQE
        vqe_results = self.vqe_solver.solve_vqe(conversion_results)
        
        # Save VQE results
        vqe_results_path = self.output_dir / "vqe_results.pkl"
        with open(vqe_results_path, 'wb') as f:
            pickle.dump(vqe_results, f)
        
        # Plot optimization history
        plot_path = self.output_dir / "vqe_convergence.png"
        self.vqe_solver.plot_optimization_history(str(plot_path))
        
        self.results['step4'] = vqe_results
        
        logger.info(f"Step 4 completed successfully")
        logger.info(f"VQE success: {vqe_results['success']}")
        if vqe_results['success']:
            logger.info(f"VQE solve time: {vqe_results['solve_time']:.2f}s")
            logger.info(f"VQE iterations: {vqe_results['num_iterations']}")
            logger.info(f"VQE optimal value: {vqe_results['optimal_value']:.6f}")
            logger.info(f"VQE selected assets: {vqe_results['num_selected']}")
        
        return self.results['step4']
    
    def step5_qaoa_optimization(self, backend_type: str = "simulator", 
                               use_noise: bool = False, shots: int = 1024, 
                               reps: int = 3) -> Dict[str, Any]:
        """
        Step 5: QAOA quantum optimization
        Following QAOA methodology from research.md
        """
        logger.info("="*60)
        logger.info("STEP 5: QAOA Quantum Optimization")
        logger.info("="*60)
        
        if 'step3' not in self.results:
            raise ValueError("Must complete step 3 first")
        
        conversion_results = self.results['step3']
        
        # Initialize QAOA solver
        self.qaoa_solver = QAOAPortfolioSolver(
            backend_type=backend_type,
            use_noise=use_noise,
            shots=shots,
            reps=reps
        )
        
        # Run QAOA
        qaoa_results = self.qaoa_solver.solve_qaoa(conversion_results, warm_start=True)
        
        # Save QAOA results
        qaoa_results_path = self.output_dir / "qaoa_results.pkl"
        with open(qaoa_results_path, 'wb') as f:
            pickle.dump(qaoa_results, f)
        
        # Plot optimization history
        plot_path = self.output_dir / "qaoa_convergence.png"
        self.qaoa_solver.plot_optimization_history(str(plot_path))
        
        # Plot angle evolution
        angles_plot_path = self.output_dir / "qaoa_angles.png"
        self.qaoa_solver.plot_angle_evolution(str(angles_plot_path))
        
        self.results['step5'] = qaoa_results
        
        logger.info(f"Step 5 completed successfully")
        logger.info(f"QAOA success: {qaoa_results['success']}")
        if qaoa_results['success']:
            logger.info(f"QAOA solve time: {qaoa_results['solve_time']:.2f}s")
            logger.info(f"QAOA iterations: {qaoa_results['num_iterations']}")
            logger.info(f"QAOA optimal value: {qaoa_results['optimal_value']:.6f}")
            logger.info(f"QAOA selected assets: {qaoa_results['num_selected']}")
        
        return self.results['step5']
    
    def step6_comparison_analysis(self) -> Dict[str, Any]:
        """
        Step 6: Compare all solutions and analyze results
        Generate comprehensive comparison following research.md
        """
        logger.info("="*60)
        logger.info("STEP 6: Results Comparison and Analysis")
        logger.info("="*60)
        
        # Collect all results
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'problem_size': self.results['step1']['portfolio_vars']['n_assets'],
            'max_assets': self.results['step1']['portfolio_vars']['max_assets']
        }
        
        # Classical warm-start results (for reference)
        if 'step2' in self.results:
            warm_start = self.results['step2']['warm_start_solution']
            comparison_data['classical_warmstart'] = {
                'solver': 'GUROBI (60s warm-start)',
                'success': warm_start['success'],
                'solve_time': warm_start['solve_time'],
                'objective_value': warm_start.get('objective_value', float('inf')),
                'num_selected': warm_start.get('num_selected', 0),
                'status': warm_start.get('status', 'Unknown'),
                'note': 'Used as initial guidance for quantum algorithms'
            }
        
        # VQE results
        if 'step4' in self.results:
            vqe = self.results['step4']
            comparison_data['vqe'] = {
                'solver': 'VQE',
                'success': vqe['success'],
                'solve_time': vqe.get('solve_time', float('inf')),
                'objective_value': vqe.get('optimal_value', float('inf')),
                'qubo_energy': vqe.get('qubo_energy', float('inf')),
                'num_selected': vqe.get('num_selected', 0),
                'num_iterations': vqe.get('num_iterations', 0),
                'backend': vqe.get('backend', 'Unknown')
            }
        
        # QAOA results
        if 'step5' in self.results:
            qaoa = self.results['step5']
            comparison_data['qaoa'] = {
                'solver': 'QAOA',
                'success': qaoa['success'],
                'solve_time': qaoa.get('solve_time', float('inf')),
                'objective_value': qaoa.get('optimal_value', float('inf')),
                'qubo_energy': qaoa.get('qubo_energy', float('inf')),
                'num_selected': qaoa.get('num_selected', 0),
                'num_iterations': qaoa.get('num_iterations', 0),
                'reps': qaoa.get('reps', 0),
                'backend': qaoa.get('backend', 'Unknown')
            }
        
        self.comparison_data = comparison_data
        
        # Save comparison data
        comparison_path = self.output_dir / "comparison_results.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Generate comparison plots
        self._create_comparison_plots()
        
        # Print summary
        self._print_comparison_summary()
        
        logger.info(f"Step 6 completed successfully")
        
        return comparison_data
    
    def _create_comparison_plots(self):
        """
        Create comprehensive comparison plots
        """
        try:
            # Comparison metrics plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            solvers = []
            solve_times = []
            objectives = []
            selected_assets = []
            
            # Collect data
            for solver_name in ['classical', 'vqe', 'qaoa']:
                if solver_name in self.comparison_data and self.comparison_data[solver_name]['success']:
                    data = self.comparison_data[solver_name]
                    solvers.append(solver_name.upper())
                    solve_times.append(data['solve_time'])
                    objectives.append(data['objective_value'])
                    selected_assets.append(data['num_selected'])
            
            if len(solvers) > 0:
                # Solve time comparison
                ax1.bar(solvers, solve_times, color=['blue', 'red', 'green'][:len(solvers)])
                ax1.set_ylabel('Solve Time (s)')
                ax1.set_title('Solver Performance: Time')
                ax1.set_yscale('log')
                
                # Objective value comparison
                ax2.bar(solvers, objectives, color=['blue', 'red', 'green'][:len(solvers)])
                ax2.set_ylabel('Objective Value')
                ax2.set_title('Solver Performance: Objective')
                
                # Number of selected assets
                ax3.bar(solvers, selected_assets, color=['blue', 'red', 'green'][:len(solvers)])
                ax3.set_ylabel('Number of Assets')
                ax3.set_title('Selected Assets Count')
                
                # Speedup comparison (if classical baseline exists)
                if 'classical' in self.comparison_data and len(solvers) > 1:
                    classical_time = self.comparison_data['classical']['solve_time']
                    speedups = [classical_time / t if t > 0 else 0 for t in solve_times[1:]]
                    quantum_solvers = solvers[1:]
                    
                    ax4.bar(quantum_solvers, speedups, color=['red', 'green'][:len(quantum_solvers)])
                    ax4.set_ylabel('Speedup Factor')
                    ax4.set_title('Quantum Speedup vs Classical')
                    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.7)
                else:
                    ax4.text(0.5, 0.5, 'No speedup\ncomparison\navailable', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Speedup Analysis')
            
            plt.tight_layout()
            comparison_plot_path = self.output_dir / "comparison_metrics.png"
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Saved comparison plot to {comparison_plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating comparison plots: {e}")
    
    def _print_comparison_summary(self):
        """
        Print a comprehensive comparison summary
        """
        print("\n" + "="*80)
        print("WISER PORTFOLIO OPTIMIZATION - RESULTS SUMMARY")
        print("="*80)
        
        print(f"Problem Size: {self.comparison_data['problem_size']} assets")
        print(f"Max Assets Constraint: {self.comparison_data['max_assets']}")
        print(f"Timestamp: {self.comparison_data['timestamp']}")
        
        print("\nSOLVER COMPARISON:")
        print("-" * 80)
        print(f"{'Solver':<10} {'Success':<8} {'Time(s)':<10} {'Objective':<15} {'Assets':<8} {'Notes':<20}")
        print("-" * 80)
        
        for solver_name in ['classical', 'vqe', 'qaoa']:
            if solver_name in self.comparison_data:
                data = self.comparison_data[solver_name]
                success = "✓" if data['success'] else "✗"
                time_str = f"{data['solve_time']:.2f}" if data['success'] else "N/A"
                obj_str = f"{data['objective_value']:.6f}" if data['success'] else "N/A"
                assets_str = f"{data['num_selected']}" if data['success'] else "N/A"
                
                notes = ""
                if solver_name == 'classical':
                    notes = data.get('status', '')
                elif solver_name in ['vqe', 'qaoa']:
                    notes = f"{data.get('backend', 'Unknown')}"
                
                print(f"{solver_name.upper():<10} {success:<8} {time_str:<10} {obj_str:<15} {assets_str:<8} {notes:<20}")
        
        # Calculate speedups and approximation ratios
        classical_data = self.comparison_data.get('classical')
        if classical_data and classical_data['success']:
            print("\nQUANTUM ALGORITHM ANALYSIS:")
            print("-" * 50)
            
            classical_time = classical_data['solve_time']
            classical_obj = classical_data['objective_value']
            
            for solver_name in ['vqe', 'qaoa']:
                if solver_name in self.comparison_data and self.comparison_data[solver_name]['success']:
                    quantum_data = self.comparison_data[solver_name]
                    speedup = classical_time / quantum_data['solve_time'] if quantum_data['solve_time'] > 0 else 0
                    approx_ratio = quantum_data['objective_value'] / classical_obj if classical_obj != 0 else float('inf')
                    
                    print(f"{solver_name.upper()}:")
                    print(f"  Speedup: {speedup:.2f}x")
                    print(f"  Approximation ratio: {approx_ratio:.4f}")
                    print(f"  Iterations: {quantum_data.get('num_iterations', 'N/A')}")
        
        print("\n" + "="*80)
    
    def run_full_pipeline(self, quantum_backend: str = "simulator", 
                         use_noise: bool = False, shots: int = 1024,
                         classical_time_limit: int = 300, qaoa_reps: int = 3):
        """
        Run the complete WISER optimization pipeline
        
        Args:
            quantum_backend: 'simulator', 'ibm_hardware', or 'ibm_simulator'
            use_noise: Add noise to quantum simulation
            shots: Number of quantum shots
            classical_time_limit: Classical solver time limit
            qaoa_reps: Number of QAOA layers
        """
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("STARTING WISER PORTFOLIO OPTIMIZATION PIPELINE")
        logger.info("="*80)
        logger.info(f"Quantum backend: {quantum_backend}")
        logger.info(f"Use noise: {use_noise}")
        logger.info(f"Shots: {shots}")
        logger.info(f"Classical time limit: {classical_time_limit}s")
        logger.info(f"QAOA layers: {qaoa_reps}")
        
        try:
            # Step 1: Data processing
            self.step1_load_and_process_data()
            
            # Step 2: Classical baseline
            self.step2_classical_baseline(time_limit=classical_time_limit)
            
            # Step 3: Quantum conversion
            self.step3_quantum_conversion()
            
            # Step 4: VQE optimization
            self.step4_vqe_optimization(
                backend_type=quantum_backend,
                use_noise=use_noise,
                shots=shots
            )
            
            # Step 5: QAOA optimization
            self.step5_qaoa_optimization(
                backend_type=quantum_backend,
                use_noise=use_noise,
                shots=shots,
                reps=qaoa_reps
            )
            
            # Step 6: Comparison analysis
            self.step6_comparison_analysis()
            
            total_time = time.time() - start_time
            
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {total_time:.2f}s")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error("Check the logs for detailed error information")
            return False
    
    def save_all_results(self):
        """
        Save all results to files
        """
        results_path = self.output_dir / "all_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"All results saved to {results_path}")

def main():
    """
    Main execution function
    """
    # Initialize optimizer
    optimizer = WISERPortfolioOptimizer()
    
    # Run full pipeline with simulator first
    success = optimizer.run_full_pipeline(
        quantum_backend="simulator",
        use_noise=False,
        shots=1024,
        classical_time_limit=60,  # Shorter for demo
        qaoa_reps=2               # Fewer layers for demo
    )
    
    if success:
        optimizer.save_all_results()
        print("\n Portfolio Optimization completed successfully!")
        print(f" Check results in: {optimizer.output_dir}")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
    
    return optimizer

if __name__ == "__main__":
    main()
