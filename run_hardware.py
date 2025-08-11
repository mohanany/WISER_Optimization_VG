

import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from main_runner import WISERPortfolioOptimizer

def main():
    print("🚀 WISER Portfolio Optimization - Hardware Runner")
    print("=" * 60)
    print(f"📅 Session: {datetime.now()}")
    
   
    config = {
        'assets': 6,           
        'max_assets': 4,          
        'target_return': 0.01,
        'risk_aversion': 1.0,
        'vqe_iterations': 30,    
        'qaoa_reps': 1,         
        'shots': 512,            
        'use_noise': False,      
        'backend_type': 'ibm_hardware'  
    }
    
    print("\n⚙️ Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
   
    if not os.getenv('IBM_API_TOKEN'):
        print("\n⚠️ IBM_API_TOKEN not found!")
        print("Set your IBM Quantum token:")
        print("   export IBM_API_TOKEN='your_token_here'")
        print("Or run: python setup_hardware.py --token YOUR_TOKEN")
        return False
    
    # إنشاء المحسن
    results_dir = f"hardware_results_{datetime.now().strftime('%Y%m%d_%H%M')}"
    optimizer = WISERPortfolioOptimizer(
        data_path="../data/1/",
        output_dir=results_dir
    )
    
    print(f"\n📁 Results will be saved to: {results_dir}")
    print("\n🎯 Starting optimization...")
    
    try:
      
        success = optimizer.run_full_pipeline(
            num_assets=config['assets'],
            max_assets=config['max_assets'],
            target_return=config['target_return'],
            risk_aversion=config['risk_aversion'],
            quantum_backend=config['backend_type'],
            use_noise=config['use_noise'],
            shots=config['shots'],
            vqe_maxiter=config['vqe_iterations'],
            qaoa_reps=config['qaoa_reps'],
            methods=['classical', 'vqe', 'qaoa']
        )
        
        if success:
            print("\n✅ Hardware optimization completed successfully!")
            print(f"📁 All results saved to: {results_dir}")
            
            # عرض ملخص النتائج
            if optimizer.comparison_data:
                print("\n📊 Results Summary:")
                data = optimizer.comparison_data
                
                for method in ['classical', 'vqe', 'qaoa']:
                    if method in data and data[method].get('success'):
                        result = data[method]
                        print(f"   {method.upper()}: {result.get('objective_value', 'N/A'):.6f} "
                              f"({result.get('solve_time', 0):.1f}s)")
            
            return True
        else:
            print("\n❌ Optimization failed - check logs")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Check your IBM Quantum credentials and connection")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
