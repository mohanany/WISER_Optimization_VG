#!/usr/bin/env python3
"""
Setup script for WISER Portfolio Optimization Challenge
Installs dependencies and validates the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"{'='*50}")
    print(f"🔧 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version OK")
    return True

def install_dependencies():
    """Install required packages"""
    packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "openpyxl>=3.0.0",
        "python-dotenv>=0.19.0",
        "jupyter>=1.0.0"
    ]
    
    print("📦 Installing basic dependencies...")
    for package in packages:
        success = run_command(
            f"{sys.executable} -m pip install '{package}'",
            f"Installing {package.split('>=')[0]}"
        )
        if not success:
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Try to install quantum packages (may fail if not available)
    quantum_packages = [
        "qiskit>=1.0.0",
        "qiskit-aer>=0.13.0",
    ]
    
    print("\n🔬 Installing quantum packages...")
    for package in quantum_packages:
        success = run_command(
            f"{sys.executable} -m pip install '{package}'",
            f"Installing {package.split('>=')[0]}"
        )
        if not success:
            print(f"⚠️  {package} installation failed - will use fallbacks")
    
    # Optional packages
    optional_packages = [
        "qiskit-optimization",
        "qiskit-ibm-runtime",
        "gurobipy"
    ]
    
    print("\n🎯 Installing optional packages...")
    for package in optional_packages:
        success = run_command(
            f"{sys.executable} -m pip install '{package}'",
            f"Installing {package} (optional)"
        )
        if not success:
            print(f"ℹ️  {package} not installed - fallbacks will be used")

def test_imports():
    """Test critical imports"""
    print("\n" + "="*50)
    print("🧪 TESTING IMPORTS")
    print("="*50)
    
    test_modules = [
        ("numpy", "np", True),
        ("pandas", "pd", True),
        ("matplotlib.pyplot", "plt", True),
        ("scipy", "", True),
        ("openpyxl", "", True),
        ("qiskit", "", False),
        ("gurobipy", "gp", False),
    ]
    
    results = {}
    
    for module, alias, required in test_modules:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            
            print(f"✅ {module} - OK")
            results[module] = True
            
        except ImportError as e:
            status = "❌ REQUIRED" if required else "⚠️  OPTIONAL"
            print(f"{status} {module} - {str(e)[:50]}...")
            results[module] = False
    
    return results

def test_data_access():
    """Test access to data files"""
    print("\n" + "="*50)
    print("📊 TESTING DATA ACCESS")
    print("="*50)
    
    data_dir = Path("../data/1/")
    required_files = [
        "data_assets_dump_partial.xlsx",
        "data_assets_dictionary.xlsx"
    ]
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir.absolute()}")
        return False
    
    print(f"✅ Data directory found: {data_dir.absolute()}")
    
    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_name} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_name} - NOT FOUND")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("These are required for the portfolio optimization")
        return False
    
    return True

def test_solution_modules():
    """Test our solution modules"""
    print("\n" + "="*50)
    print("🔬 TESTING SOLUTION MODULES")
    print("="*50)
    
    modules = [
        "data_processor",
        "classical_solver", 
        "qubo_converter",
        "vqe_solver",
        "qaoa_solver",
        "main_runner"
    ]
    
    success_count = 0
    
    for module in modules:
        try:
            exec(f"import {module}")
            print(f"✅ {module}.py - OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}.py - {str(e)[:50]}...")
    
    print(f"\n📊 Module test results: {success_count}/{len(modules)} successful")
    return success_count == len(modules)

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path(".env")
    template_file = Path(".env.template")
    
    if not env_file.exists() and template_file.exists():
        print("\n🔧 Creating .env file from template...")
        with open(template_file, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ .env file created")
        print("⚠️  Remember to add your actual IBM Quantum credentials!")
        return True
    
    return False

def main():
    """Main setup function"""
    print("🚀 WISER Portfolio Optimization Challenge - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python():
        return False
    
    # Install dependencies
    install_dependencies()
    
    # Test imports
    import_results = test_imports()
    
    # Test data access
    data_ok = test_data_access()
    
    # Test solution modules
    modules_ok = test_solution_modules()
    
    # Create .env file
    create_env_file()
    
    # Final summary
    print("\n" + "="*60)
    print("📋 SETUP SUMMARY")
    print("="*60)
    
    print(f"✅ Python: OK")
    print(f"{'✅' if import_results.get('numpy') else '❌'} NumPy: {'OK' if import_results.get('numpy') else 'FAILED'}")
    print(f"{'✅' if import_results.get('pandas') else '❌'} Pandas: {'OK' if import_results.get('pandas') else 'FAILED'}")
    print(f"{'✅' if import_results.get('qiskit') else '⚠️ '} Qiskit: {'OK' if import_results.get('qiskit') else 'NOT AVAILABLE'}")
    print(f"{'✅' if import_results.get('gurobipy') else '⚠️ '} GUROBI: {'OK' if import_results.get('gurobipy') else 'NOT AVAILABLE'}")
    print(f"{'✅' if data_ok else '❌'} Data files: {'OK' if data_ok else 'MISSING'}")
    print(f"{'✅' if modules_ok else '❌'} Solution modules: {'OK' if modules_ok else 'FAILED'}")
    
    # Final recommendation
    required_ok = (import_results.get('numpy') and 
                   import_results.get('pandas') and 
                   data_ok and modules_ok)
    
    if required_ok:
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("✅ Ready to run WISER Portfolio Optimization")
        print("\n🚀 Next steps:")
        print("  1. Run: python3 main_runner.py")
        print("  2. Or: jupyter notebook WISER_Complete_Demo.ipynb")
        print("  3. For IBM hardware: set credentials in .env")
        return True
    else:
        print("\n❌ SETUP INCOMPLETE")
        print("⚠️  Some required components are missing")
        print("Please resolve the issues above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
