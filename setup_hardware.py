#!/usr/bin/env python3
"""
🚀 WISER Hardware Setup Script
إعداد سريع لتشغيل الخوارزميات الكمية على الأجهزة الحقيقية

الاستخدام:
    python setup_hardware.py --token YOUR_IBM_TOKEN
    
أو تعديل متغيرات البيئة:
    export IBM_API_TOKEN="your_token_here"
    python setup_hardware.py
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """إعداد البيئة والحزم المطلوبة"""
    print("🔧 Setting up quantum environment...")
    
    try:
        import qiskit
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_algorithms import VQE, QAOA
        print(f"✅ Qiskit {qiskit.__version__} ready")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("📦 Please install required packages:")
        print("    pip install qiskit qiskit-ibm-runtime qiskit-algorithms qiskit-aer")
        return False

def test_ibm_connection(token=None):
    """اختبار الاتصال بـ IBM Quantum"""
    print("\n🔗 Testing IBM Quantum connection...")
    
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        
        if token:
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            # Save token for future use
            service.save_account(channel="ibm_quantum", token=token, overwrite=True)
            print("💾 Token saved for future sessions")
        else:
            service = QiskitRuntimeService()
        
        # List available backends
        backends = service.backends(operational=True, simulator=False)
        
        if backends:
            print(f"✅ Connected successfully! Found {len(backends)} available backends:")
            
            # Show top 3 backends with least queue
            backend_info = []
            for backend in backends:
                try:
                    status = service.backend(backend.name).status()
                    backend_info.append({
                        'name': backend.name,
                        'qubits': backend.num_qubits,
                        'pending': status.pending_jobs
                    })
                except:
                    continue
            
            backend_info.sort(key=lambda x: x['pending'])
            
            print("\n🖥️ Best backends (by queue length):")
            for info in backend_info[:3]:
                print(f"   {info['name']}: {info['qubits']} qubits, {info['pending']} pending jobs")
                
            return service, backend_info[0]['name']
        else:
            print("❌ No operational backends found")
            return None, None
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔑 To connect:")
        print("1. Get your token from: https://quantum-computing.ibm.com/")
        print("2. Run: python setup_hardware.py --token YOUR_TOKEN")
        print("3. Or set environment variable: export IBM_API_TOKEN='your_token'")
        return None, None

def run_quick_test(service, backend_name):
    """تشغيل اختبار سريع على الجهاز الحقيقي"""
    print(f"\n🧪 Running quick test on {backend_name}...")
    
    try:
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.circuit.library import RealAmplitudes
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_ibm_runtime import Estimator
        
        # Simple 2-qubit problem
        hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 0.5), ("IZ", 0.5)])
        ansatz = RealAmplitudes(2, reps=1)
        optimizer = COBYLA(maxiter=10)  # Very few iterations for testing
        
        backend = service.backend(backend_name)
        estimator = Estimator(backend=backend)
        
        vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer)
        
        print(f"   Running VQE with {ansatz.num_parameters} parameters...")
        print(f"   ⚠️ This will consume quantum credits!")
        
        # Ask for confirmation
        confirm = input("   Continue with hardware test? (y/N): ").lower().strip()
        if confirm != 'y':
            print("   Test skipped")
            return False
        
        start_time = datetime.now()
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Hardware test successful!")
        print(f"   Eigenvalue: {result.eigenvalue:.6f}")
        print(f"   Execution time: {execution_time:.1f} seconds")
        print(f"   Backend: {backend_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        return False

def create_config_file(backend_name):
    """إنشاء ملف إعدادات للتشغيل"""
    config = f"""# WISER Hardware Configuration
# Generated: {datetime.now()}

BACKEND_NAME = "{backend_name}"
ASSETS_COUNT = 6        # Start small for hardware
VQE_ITERATIONS = 30     # Limit iterations to save credits
QAOA_REPS = 1          # Simple QAOA depth
RISK_AVERSION = 1.0
SAVE_RESULTS = True
RESULTS_FILE = "hardware_results_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"

# Hardware optimization tips:
# - Start with small problems (4-8 assets)
# - Use fewer optimizer iterations
# - Monitor your credit usage
# - Save results frequently
"""
    
    with open('hardware_config.py', 'w') as f:
        f.write(config)
    
    print(f"📝 Configuration saved to: hardware_config.py")

def main():
    parser = argparse.ArgumentParser(description='WISER Quantum Hardware Setup')
    parser.add_argument('--token', type=str, help='IBM Quantum API token')
    parser.add_argument('--test', action='store_true', help='Run hardware test')
    parser.add_argument('--config-only', action='store_true', help='Only create config file')
    
    args = parser.parse_args()
    
    print("🚀 WISER Quantum Portfolio Optimization - Hardware Setup")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Check for token
    token = args.token or os.getenv('IBM_API_TOKEN')
    
    if args.config_only:
        create_config_file("ibm_brisbane")  # Default backend
        print("✅ Config file created. Edit as needed.")
        return
    
    # Test IBM connection
    service, best_backend = test_ibm_connection(token)
    
    if not service:
        print("\n⚠️ Cannot connect to IBM Quantum")
        print("You can still create config files and run simulator tests")
        create_config_file("aer_simulator")
        return
    
    # Create config file
    create_config_file(best_backend)
    
    # Run test if requested
    if args.test:
        success = run_quick_test(service, best_backend)
        if success:
            print("\n🎉 Hardware setup complete!")
            print("✅ Ready to run quantum portfolio optimization")
        else:
            print("\n⚠️ Hardware test failed, but setup is complete")
            print("You can still use the configuration for simulator runs")
    else:
        print("\n✅ Setup complete!")
        print("🧪 Add --test flag to run a quick hardware test")
    
    print(f"\n📂 Next steps:")
    print(f"1. Open quantum_hardware.ipynb in Jupyter")
    print(f"2. Or run the main optimization script")
    print(f"3. Monitor your credit usage at: https://quantum-computing.ibm.com/")

if __name__ == "__main__":
    main()
