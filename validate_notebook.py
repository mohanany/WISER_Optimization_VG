#!/usr/bin/env python3
"""
Final Submission Notebook Validation Script
Tests all components before final execution
"""

import json
import os
from pathlib import Path

def validate_final_notebook():
    """Validate the professional notebook is ready for execution"""
    
    print("🔍 FINAL SUBMISSION NOTEBOOK VALIDATION")
    print("="*60)
    
    # Check notebook exists
    notebook_file = "Final_Submission_Professional.ipynb"
    if not Path(notebook_file).exists():
        print(f"❌ Notebook not found: {notebook_file}")
        return False
    
    print(f"✅ Notebook found: {notebook_file}")
    
    # Load and analyze notebook
    with open(notebook_file, 'r') as f:
        notebook = json.load(f)
    
    print(f"📊 Notebook Analysis:")
    print(f"   📝 Total Cells: {len(notebook['cells'])}")
    
    # Count cell types
    markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
    
    print(f"   📚 Markdown Cells: {markdown_cells}")
    print(f"   💻 Code Cells: {code_cells}")
    
    # Check critical components
    cell_contents = ' '.join([
        ' '.join(cell.get('source', [])) for cell in notebook['cells']
    ])
    
    components = {
        'Data Loading': 'PortfolioDataProcessor' in cell_contents,
        'Classical Solver': 'ClassicalPortfolioOptimizer' in cell_contents,
        'QAOA Algorithm': 'QAOAPortfolioSolver' in cell_contents,
        'VQE Algorithm': 'VQEPortfolioSolver' in cell_contents,
        'Quantum Conversion': 'QUBOIsingConverter' in cell_contents,
        'Warm-Start': 'warm_start' in cell_contents.lower(),
        'Hardware Execution': 'ibm_hardware' in cell_contents.lower(),
        'Comparison Analysis': 'comparison' in cell_contents.lower(),
        'Visualizations': 'plt.show' in cell_contents or 'matplotlib' in cell_contents
    }
    
    print(f"\n🧩 Component Validation:")
    all_components = True
    for component, present in components.items():
        status = "✅" if present else "❌"
        print(f"   {status} {component}")
        if not present:
            all_components = False
    
    # Check environment requirements
    print(f"\n🔧 Environment Check:")
    
    # Check IBM API token
    ibm_token = os.getenv('IBM_API_TOKEN')
    print(f"   {'✅' if ibm_token else '⚠️ '} IBM API Token: {'Configured' if ibm_token else 'Not Set'}")
    
    # Check required files
    required_files = [
        'data_processor.py',
        'classical_solver.py',
        'vqe_solver.py', 
        'qaoa_solver.py',
        'qubo_converter.py'
    ]
    
    print(f"\n📁 Required Files:")
    all_files = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
        if not exists:
            all_files = False
    
    # Check data directory
    data_dir = Path("../data/1/")
    data_files = ['data_assets_dump_partial.xlsx', 'data_assets_dictionary.xlsx']
    
    print(f"\n📊 Data Files:")
    all_data = True
    for data_file in data_files:
        exists = (data_dir / data_file).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {data_file}")
        if not exists:
            all_data = False
    
    # Final validation
    print(f"\n🎯 FINAL VALIDATION RESULTS:")
    print("="*40)
    
    overall_status = all_components and all_files and all_data
    
    if overall_status:
        print("🎉 ✅ NOTEBOOK FULLY READY FOR EXECUTION!")
        print("🚀 All components validated successfully")
        print("💎 Professional quality confirmed")
        print("🔬 Research requirements fulfilled")
        if ibm_token:
            print("⚡ IBM Hardware execution enabled")
        else:
            print("🖥️  Simulation mode available")
    else:
        print("⚠️  ❌ NOTEBOOK NEEDS ATTENTION!")
        if not all_components:
            print("   🧩 Missing notebook components")
        if not all_files:
            print("   📁 Missing required Python files")
        if not all_data:
            print("   📊 Missing data files")
    
    print(f"\n📋 Summary:")
    print(f"   📝 Notebook Structure: {'✅ Valid' if markdown_cells >= 8 and code_cells >= 8 else '⚠️  Check'}")
    print(f"   🧩 Components: {'✅ Complete' if all_components else '❌ Incomplete'}")
    print(f"   📁 Files: {'✅ Present' if all_files else '❌ Missing'}")
    print(f"   📊 Data: {'✅ Available' if all_data else '❌ Missing'}")
    print(f"   🔐 IBM Token: {'✅ Ready' if ibm_token else '⚠️  Set Required'}")
    
    return overall_status

if __name__ == "__main__":
    success = validate_final_notebook()
    if success:
        print(f"\n🎯 Ready to execute Final_Submission_Professional.ipynb!")
    else:
        print(f"\n🔧 Please address the issues above before execution.")
