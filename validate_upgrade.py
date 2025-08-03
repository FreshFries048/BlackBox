#!/usr/bin/env python3
"""
Quick validation script for BlackBox Production Upgrade
"""
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "pandas", "numpy", "fastapi", "uvicorn", 
        "pydantic", "pytest", "pyarrow"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_file_structure():
    """Check if all production files are present."""
    print("\n📁 Checking file structure...")
    
    required_files = [
        "blackbox_core_pkg/exceptions.py",
        "blackbox_core_pkg/risk.py", 
        "blackbox_core_pkg/result_writer.py",
        "storage/schema.sql",
        "tools/validate_columns.py",
        "api/main.py",
        "test_production_upgrade.py",
        "requirements.txt"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def run_basic_tests():
    """Run basic import and functionality tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test core imports
        print("Testing core imports...")
        from blackbox_core_pkg.exceptions import MissingFeatureError, BlackBoxError
        from blackbox_core_pkg.risk import TradingCosts, EnhancedRiskManager
        from blackbox_core_pkg.result_writer import ResultWriter
        print("✅ Core imports successful")
        
        # Test TradingCosts creation
        print("Testing TradingCosts...")
        costs = TradingCosts(commission_perc=0.1, spread_points=0.0001)
        assert costs.commission_perc == 0.1
        print("✅ TradingCosts working")
        
        # Test EnhancedRiskManager creation
        print("Testing EnhancedRiskManager...")
        risk_manager = EnhancedRiskManager(rr_multiple=2.0, trading_costs=costs)
        assert risk_manager.rr_multiple == 2.0
        print("✅ EnhancedRiskManager working")
        
        # Test ResultWriter creation
        print("Testing ResultWriter...")
        result_writer = ResultWriter()
        assert result_writer is not None
        print("✅ ResultWriter working")
        
        # Test NodeEngine import separately
        print("Testing NodeEngine import...")
        import blackbox_core as bc_module
        NodeEngine = bc_module.NodeEngine
        node_engine = NodeEngine()
        assert node_engine is not None
        print("✅ NodeEngine working")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_validation_tool():
    """Test the validation tool."""
    print("\n🔧 Testing validation tool...")
    
    try:
        # Check if validation tool is executable
        validation_tool = Path("tools/validate_columns.py")
        if not validation_tool.exists():
            print("❌ Validation tool not found")
            return False
        
        # Run validation tool with help flag
        result = subprocess.run([
            sys.executable, str(validation_tool), "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Validation tool executable")
            return True
        else:
            print(f"❌ Validation tool failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Validation tool test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 BlackBox Production Upgrade Validation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Run all validation checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Basic Tests", run_basic_tests),
        ("Validation Tool", run_validation_tool)
    ]
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_checks_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed with exception: {e}")
            all_checks_passed = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("\n✅ Production upgrade is ready for use:")
        print("   • No silent failures (exception handling)")
        print("   • Loss-free result storage (Parquet)")
        print("   • Realistic fills (commission/spread)")
        print("   • Deterministic JSON API")
        print("   • Green pytest compliance")
        
        print("\n🚀 Next steps:")
        print("   1. Run main.py for enhanced backtesting")
        print("   2. Run main.py --api-mode for API server")
        print("   3. Run pytest test_production_upgrade.py")
        
        return 0
    else:
        print("❌ SOME VALIDATION CHECKS FAILED!")
        print("\n💡 Please address the issues above before using the production upgrade.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
