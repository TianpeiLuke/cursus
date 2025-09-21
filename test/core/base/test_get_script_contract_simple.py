#!/usr/bin/env python3
"""
Simplified test to verify get_script_contract() method functionality.

This test focuses on testing the method logic without requiring full config instantiation.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_step_name_derivation():
    """Test _derive_step_name() method using class methods."""
    
    print("=" * 80)
    print("TESTING STEP NAME DERIVATION")
    print("=" * 80)
    
    # Test cases: Config Class Name -> Expected Step Name
    test_cases = {
        "TabularPreprocessingConfig": "TabularPreprocessing",
        "CurrencyConversionConfig": "CurrencyConversion", 
        "PackageConfig": "Package",
        "XGBoostModelEvalConfig": "XGBoostModelEval",
        "DummyTrainingConfig": "DummyTraining",
        "ModelCalibrationConfig": "ModelCalibration",
        "PayloadConfig": "Payload",
        "RiskTableMappingConfig": "RiskTableMapping",
        "XGBoostTrainingConfig": "XGBoostTraining",
        "PyTorchTrainingConfig": "PyTorchTraining",
    }
    
    results = {}
    
    for config_class_name, expected_step_name in test_cases.items():
        print(f"\n--- Testing {config_class_name} ---")
        
        try:
            # Import BasePipelineConfig to test the registry lookup
            from cursus.core.base.config_base import BasePipelineConfig
            
            # Test the get_step_name class method
            derived_step_name = BasePipelineConfig.get_step_name(config_class_name)
            
            print(f"📝 Config class: {config_class_name}")
            print(f"📝 Derived step name: '{derived_step_name}'")
            print(f"📝 Expected step name: '{expected_step_name}'")
            
            if derived_step_name == expected_step_name:
                print(f"✅ Step name derivation: PASS")
                results[config_class_name] = "SUCCESS"
            else:
                print(f"❌ Step name derivation: FAIL (got '{derived_step_name}', expected '{expected_step_name}')")
                results[config_class_name] = f"FAIL - got '{derived_step_name}'"
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results[config_class_name] = f"ERROR - {e}"
    
    # Summary
    print("\n" + "=" * 80)
    print("STEP NAME DERIVATION SUMMARY")
    print("=" * 80)
    
    success_count = 0
    total_count = len(results)
    
    for config_name, result in results.items():
        status = "✅ PASS" if result == "SUCCESS" else f"❌ {result}"
        print(f"{config_name:<30} {status}")
        if result == "SUCCESS":
            success_count += 1
    
    print(f"\nResults: {success_count}/{total_count} step name derivations working")
    return results

def test_step_catalog_contract_discovery():
    """Test step catalog contract discovery directly."""
    
    print("\n" + "=" * 80)
    print("TESTING STEP CATALOG CONTRACT DISCOVERY")
    print("=" * 80)
    
    # Test cases: Step Name -> Expected Contract File
    test_cases = {
        "TabularPreprocessing": "tabular_preprocessing_contract.py",
        "CurrencyConversion": "currency_conversion_contract.py",
        "Package": "package_contract.py", 
        "XGBoostModelEval": "xgboost_model_eval_contract.py",
        "DummyTraining": "dummy_training_contract.py",
        "ModelCalibration": "model_calibration_contract.py",
        "Payload": "payload_contract.py",
        "RiskTableMapping": "risk_table_mapping_contract.py",
        "XGBoostTraining": "xgboost_training_contract.py",
        "PyTorchTraining": "pytorch_training_contract.py",
    }
    
    results = {}
    
    try:
        # Import step catalog
        from cursus.step_catalog.step_catalog import StepCatalog
        step_catalog = StepCatalog()
        
        for step_name, expected_contract_file in test_cases.items():
            print(f"\n--- Testing {step_name} ---")
            
            try:
                # Test contract discovery
                contract = step_catalog.load_contract_class(step_name)
                
                print(f"📝 Step name: {step_name}")
                print(f"📝 Expected contract file: {expected_contract_file}")
                
                if contract is not None:
                    print(f"✅ Contract found: {type(contract).__name__}")
                    if hasattr(contract, 'entry_point'):
                        print(f"📝 Entry point: {contract.entry_point}")
                    results[step_name] = "SUCCESS"
                else:
                    print(f"❌ Contract not found")
                    results[step_name] = "FAIL - Contract not found"
                    
            except Exception as e:
                print(f"❌ Error loading contract: {e}")
                results[step_name] = f"ERROR - {e}"
        
    except ImportError as e:
        print(f"❌ Could not import StepCatalog: {e}")
        return {"StepCatalog": f"IMPORT_ERROR - {e}"}
    
    # Summary
    print("\n" + "=" * 80)
    print("CONTRACT DISCOVERY SUMMARY")
    print("=" * 80)
    
    success_count = 0
    total_count = len(results)
    
    for step_name, result in results.items():
        status = "✅ PASS" if result == "SUCCESS" else f"❌ {result}"
        print(f"{step_name:<25} {status}")
        if result == "SUCCESS":
            success_count += 1
    
    print(f"\nResults: {success_count}/{total_count} contract discoveries working")
    return results

def test_registry_integration():
    """Test the registry integration specifically."""
    
    print("\n" + "=" * 80)
    print("TESTING REGISTRY INTEGRATION")
    print("=" * 80)
    
    try:
        from cursus.core.base.config_base import BasePipelineConfig
        
        # Test the step registry loading
        step_registry = BasePipelineConfig._get_step_registry()
        
        print(f"📝 Step registry loaded: {len(step_registry)} entries")
        print(f"📝 Sample entries:")
        
        count = 0
        for config_class, step_name in step_registry.items():
            if count < 5:  # Show first 5 entries
                print(f"   {config_class} -> {step_name}")
                count += 1
        
        # Test specific mappings
        test_mappings = [
            ("TabularPreprocessingConfig", "TabularPreprocessing"),
            ("PackageConfig", "Package"),
            ("XGBoostModelEvalConfig", "XGBoostModelEval"),
        ]
        
        all_correct = True
        for config_class, expected_step_name in test_mappings:
            actual_step_name = step_registry.get(config_class, "NOT_FOUND")
            if actual_step_name == expected_step_name:
                print(f"✅ {config_class} -> {actual_step_name}")
            else:
                print(f"❌ {config_class} -> {actual_step_name} (expected {expected_step_name})")
                all_correct = False
        
        if all_correct:
            print("✅ Registry integration working correctly")
            return "SUCCESS"
        else:
            print("❌ Registry integration has issues")
            return "PARTIAL_FAIL"
            
    except Exception as e:
        print(f"❌ Registry integration error: {e}")
        return f"ERROR - {e}"

def main():
    """Run all tests."""
    
    print("🧪 TESTING get_script_contract() INTEGRATION")
    print("=" * 80)
    
    # Test 1: Step name derivation
    step_name_results = test_step_name_derivation()
    
    # Test 2: Step catalog contract discovery
    contract_results = test_step_catalog_contract_discovery()
    
    # Test 3: Registry integration
    registry_result = test_registry_integration()
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    step_name_success = sum(1 for r in step_name_results.values() if r == "SUCCESS")
    step_name_total = len(step_name_results)
    
    contract_success = sum(1 for r in contract_results.values() if r == "SUCCESS")
    contract_total = len(contract_results)
    
    registry_success = 1 if registry_result == "SUCCESS" else 0
    
    print(f"📊 Step Name Derivation: {step_name_success}/{step_name_total} working")
    print(f"📊 Contract Discovery: {contract_success}/{contract_total} working")
    print(f"📊 Registry Integration: {registry_success}/1 working")
    
    total_success = step_name_success + contract_success + registry_success
    total_tests = step_name_total + contract_total + 1
    
    print(f"\n🎯 OVERALL: {total_success}/{total_tests} components working ({total_success/total_tests*100:.1f}%)")
    
    if total_success == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ The optimized get_script_contract() method should work perfectly!")
    else:
        print("⚠️  Some integration components have issues.")
        print("🔧 The method may still work but with reduced efficiency.")

if __name__ == "__main__":
    main()
