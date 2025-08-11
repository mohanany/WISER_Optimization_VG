"""
IBM Hardware Runner for WISER Portfolio Optimization
Dedicated script for running on real IBM Quantum hardware
"""

import numpy as np
import logging
import time
import pickle
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

# Import our modules
from main_runner import WISERPortfolioOptimizer
from data_processor import PortfolioDataProcessor
from qubo_converter import QUBOIsingConverter
from vqe_solver import VQEPortfolioSolver
from qaoa_solver import QAOAPortfolioSolver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ibm_hardware_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IBMHardwareRunner:
    """
    Specialized runner for IBM Quantum hardware experiments
    """
    
    def __init__(self, output_dir: str = "ibm_results/"):
        load_dotenv()  # Load environment variables
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check IBM credentials
        self.check_ibm_credentials()
        
        logger.info(f"IBM Hardware Runner initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def check_ibm_credentials(self):
        """
        Check if IBM Quantum credentials are available
        """
        token = os.getenv('IBM_API_TOKEN')
        crn = os.getenv('IBM_CRN')
        instance = os.getenv('IBM_INSTANCE')
        
        if not token:
            logger.warning("IBM_API_TOKEN not found in environment")
            logger.warning("Please set your IBM Quantum API token in .env file")
            logger.warning("You can get it from: https://quantum.ibm.com/")
            return False
        
        if not crn:
            logger.info("IBM_CRN not set - using default public access")
        
        if not instance:
            logger.info("IBM_INSTANCE not set - using default: ibm-q/open/main")
            os.environ['IBM_INSTANCE'] = 'ibm-q/open/main'
        
        logger.info("✅ IBM Quantum credentials configured")
        return True
    
    def run_hardware_experiments(self, problem_size: int = 15, 
                                backends: List[str] = None,
                                shots: int = 1024,
                                qaoa_reps: int = 2):
        """
        Run experiments on IBM Quantum hardware
        
        Args:
            problem_size: Number of assets to optimize (keep small for hardware)
            backends: List of IBM backends to try
            shots: Number of quantum shots
            qaoa_reps: Number of QAOA layers
        """
        if backends is None:
            backends = ["ibm_hardware"]  # Will auto-select best available
        
        logger.info("="*80)
        logger.info("STARTING IBM QUANTUM HARDWARE EXPERIMENTS")
        logger.info("="*80)
        logger.info(f"Problem size: {problem_size} assets")
        logger.info(f"Shots: {shots}")
        logger.info(f"QAOA layers: {qaoa_reps}")
        
        # Prepare problem (use smaller size for hardware)
        optimizer = WISERPortfolioOptimizer(output_dir=str(self.output_dir))
        
        # Step 1-3: Prepare problem
        optimizer.step1_load_and_process_data()
        
        # Reduce problem size for hardware
        original_data = optimizer.results['step1']['portfolio_vars']
        if original_data['n_assets'] > problem_size:
            logger.info(f"Reducing problem size from {original_data['n_assets']} to {problem_size}")
            optimizer.data_processor.processed_data['n_assets'] = problem_size
            
            # Update all data arrays
            for key in ['asset_names', 'returns', 'risks', 'prices', 'initial_weights']:
                if key in optimizer.data_processor.processed_data:
                    optimizer.data_processor.processed_data[key] = \
                        optimizer.data_processor.processed_data[key][:problem_size]
            
            # Update max assets
            optimizer.data_processor.processed_data['max_assets'] = min(5, problem_size // 3)
            
            # Recreate problem matrices
            problem_matrices = optimizer.data_processor.create_problem_matrices()
            optimizer.results['step1']['problem_matrices'] = problem_matrices
        
        optimizer.step2_classical_baseline(time_limit=60)
        optimizer.step3_quantum_conversion()
        
        # Run hardware experiments
        hardware_results = {}
        
        for backend_type in backends:
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING ON: {backend_type}")
            logger.info(f"{'='*60}")
            
            try:
                # VQE on hardware
                logger.info("Running VQE on IBM hardware...")
                vqe_solver = VQEPortfolioSolver(
                    backend_type=backend_type,
                    use_noise=False,
                    shots=shots
                )
                
                vqe_start = time.time()
                vqe_results = vqe_solver.solve_vqe(optimizer.results['step3'])
                vqe_time = time.time() - vqe_start
                
                logger.info(f"VQE completed in {vqe_time:.2f}s")
                
                # QAOA on hardware
                logger.info("Running QAOA on IBM hardware...")
                qaoa_solver = QAOAPortfolioSolver(
                    backend_type=backend_type,
                    use_noise=False,
                    shots=shots,
                    reps=qaoa_reps
                )
                
                qaoa_start = time.time()
                qaoa_results = qaoa_solver.solve_qaoa(optimizer.results['step3'], warm_start=True)
                qaoa_time = time.time() - qaoa_start
                
                logger.info(f"QAOA completed in {qaoa_time:.2f}s")
                
                # Store results
                hardware_results[backend_type] = {
                    'vqe': vqe_results,
                    'qaoa': qaoa_results,
                    'total_time': vqe_time + qaoa_time,
                    'problem_size': problem_size,
                    'shots': shots
                }
                
                # Save individual results
                backend_dir = self.output_dir / f"hardware_{backend_type}"
                backend_dir.mkdir(exist_ok=True)
                
                with open(backend_dir / "vqe_results.pkl", 'wb') as f:
                    pickle.dump(vqe_results, f)
                
                with open(backend_dir / "qaoa_results.pkl", 'wb') as f:
                    pickle.dump(qaoa_results, f)
                
                logger.info(f"✅ Hardware experiment completed for {backend_type}")
                
            except Exception as e:
                logger.error(f"❌ Hardware experiment failed for {backend_type}: {e}")
                hardware_results[backend_type] = {
                    'error': str(e),
                    'success': False
                }
        
        # Save all hardware results
        hardware_results_path = self.output_dir / "hardware_results.pkl"
        with open(hardware_results_path, 'wb') as f:
            pickle.dump(hardware_results, f)
        
        # Generate hardware report
        self.generate_hardware_report(hardware_results, optimizer.results['step2'])
        
        logger.info("="*80)
        logger.info("IBM QUANTUM HARDWARE EXPERIMENTS COMPLETED")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*80)
        
        return hardware_results
    
    def generate_hardware_report(self, hardware_results: Dict[str, Any], 
                                classical_results: Dict[str, Any]):
        """
        Generate comprehensive report of hardware experiments
        """
        report_lines = [
            "="*80,
            "IBM QUANTUM HARDWARE EXPERIMENT REPORT",
            "="*80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Classical baseline
        classical_solution = classical_results['full_solution']
        if classical_solution['success']:
            report_lines.extend([
                "CLASSICAL BASELINE (GUROBI):",
                f"  Status: {classical_solution['status']}",
                f"  Solve time: {classical_solution['solve_time']:.2f}s",
                f"  Objective: {classical_solution['objective_value']:.6f}",
                f"  Selected assets: {classical_solution.get('num_selected', 0)}",
                ""
            ])
        
        # Hardware results
        report_lines.append("QUANTUM HARDWARE RESULTS:")
        report_lines.append("-" * 40)
        
        for backend, results in hardware_results.items():
            if results.get('error'):
                report_lines.extend([
                    f"\n{backend.upper()}: FAILED",
                    f"  Error: {results['error']}"
                ])
                continue
            
            report_lines.append(f"\n{backend.upper()}:")
            
            # VQE results
            vqe = results.get('vqe', {})
            if vqe.get('success'):
                report_lines.extend([
                    f"  VQE:",
                    f"    Success: ✅",
                    f"    Solve time: {vqe.get('solve_time', 0):.2f}s",
                    f"    Iterations: {vqe.get('num_iterations', 0)}",
                    f"    Optimal value: {vqe.get('optimal_value', float('inf')):.6f}",
                    f"    Selected assets: {vqe.get('num_selected', 0)}",
                    f"    Backend: {vqe.get('backend', 'Unknown')}"
                ])
            else:
                report_lines.append(f"  VQE: ❌ Failed")
            
            # QAOA results
            qaoa = results.get('qaoa', {})
            if qaoa.get('success'):
                report_lines.extend([
                    f"  QAOA:",
                    f"    Success: ✅",
                    f"    Solve time: {qaoa.get('solve_time', 0):.2f}s",
                    f"    Layers (p): {qaoa.get('reps', 0)}",
                    f"    Iterations: {qaoa.get('num_iterations', 0)}",
                    f"    Optimal value: {qaoa.get('optimal_value', float('inf')):.6f}",
                    f"    Selected assets: {qaoa.get('num_selected', 0)}",
                    f"    Backend: {qaoa.get('backend', 'Unknown')}"
                ])
            else:
                report_lines.append(f"  QAOA: ❌ Failed")
            
            total_time = results.get('total_time', 0)
            report_lines.append(f"  Total quantum time: {total_time:.2f}s")
        
        # Performance analysis
        if classical_solution['success']:
            report_lines.extend([
                "",
                "PERFORMANCE ANALYSIS:",
                "-" * 40
            ])
            
            classical_time = classical_solution['solve_time']
            classical_obj = classical_solution['objective_value']
            
            for backend, results in hardware_results.items():
                if results.get('error'):
                    continue
                
                report_lines.append(f"\n{backend.upper()} vs Classical:")
                
                for alg_name in ['vqe', 'qaoa']:
                    alg_results = results.get(alg_name, {})
                    if alg_results.get('success'):
                        quantum_time = alg_results.get('solve_time', 0)
                        quantum_obj = alg_results.get('optimal_value', float('inf'))
                        
                        speedup = classical_time / quantum_time if quantum_time > 0 else 0
                        approx_ratio = quantum_obj / classical_obj if classical_obj != 0 else float('inf')
                        
                        report_lines.extend([
                            f"  {alg_name.upper()}:",
                            f"    Speedup: {speedup:.2f}x",
                            f"    Approximation ratio: {approx_ratio:.4f}",
                        ])
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        # Save report
        report_path = self.output_dir / "hardware_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print report
        print('\n'.join(report_lines))
        
        logger.info(f"Hardware report saved to {report_path}")
    
    def quick_test(self, problem_size: int = 10):
        """
        Quick test on IBM hardware with minimal problem size
        """
        logger.info("Running quick IBM hardware test...")
        
        return self.run_hardware_experiments(
            problem_size=problem_size,
            backends=["ibm_hardware"],
            shots=512,  # Fewer shots for quick test
            qaoa_reps=1  # Single layer QAOA
        )

def main():
    """
    Main IBM hardware runner
    """
    runner = IBMHardwareRunner()
    
    print("\n🚀 WISER Portfolio Optimization - IBM Quantum Hardware")
    print("="*60)
    
    # Check if we should run full experiment or quick test
    choice = input("Run [F]ull experiment or [Q]uick test? (Q): ").strip().upper()
    
    if choice == 'F':
        # Full experiment
        problem_size = input("Problem size (15): ").strip()
        problem_size = int(problem_size) if problem_size else 15
        
        shots = input("Number of shots (1024): ").strip()
        shots = int(shots) if shots else 1024
        
        results = runner.run_hardware_experiments(
            problem_size=problem_size,
            shots=shots
        )
    else:
        # Quick test
        results = runner.quick_test(problem_size=8)
    
    print(f"\n✅ IBM Hardware experiments completed!")
    print(f"📁 Results saved to: {runner.output_dir}")
    
    return runner, results

if __name__ == "__main__":
    main()
