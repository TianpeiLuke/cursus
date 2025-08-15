"""
Example usage of the enhanced UniversalStepBuilderTest with scoring integration.

This script demonstrates how to use the new scoring and reporting features
that have been integrated into the universal test system.
"""

from pathlib import Path
import json

# Example usage of the enhanced UniversalStepBuilderTest
def demonstrate_enhanced_testing():
    """Demonstrate the enhanced testing capabilities."""
    
    print("🔍 Enhanced Universal Step Builder Test Demonstration")
    print("=" * 60)
    
    try:
        # Import a builder class for testing
        from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        from .universal_test import UniversalStepBuilderTest
        
        print(f"\n📋 Testing: {TabularPreprocessingStepBuilder.__name__}")
        
        # Example 1: Basic testing with scoring (new default behavior)
        print("\n1️⃣  Basic Testing with Scoring:")
        print("-" * 40)
        
        tester = UniversalStepBuilderTest(
            TabularPreprocessingStepBuilder, 
            verbose=True,  # Show detailed output
            enable_scoring=True
        )
        
        results = tester.run_all_tests()
        
        print(f"\n✅ Test completed!")
        print(f"   - Test results available in: results['test_results']")
        print(f"   - Scoring data available in: results['scoring']")
        print(f"   - Overall score: {results['scoring']['overall']['score']:.1f}/100")
        print(f"   - Rating: {results['scoring']['overall']['rating']}")
        
        # Example 2: Full reporting with structured output
        print("\n\n2️⃣  Full Reporting with Structured Output:")
        print("-" * 40)
        
        tester_full = UniversalStepBuilderTest(
            TabularPreprocessingStepBuilder,
            verbose=False,  # Suppress console output for this example
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        full_results = tester_full.run_all_tests()
        
        print(f"✅ Full report generated!")
        print(f"   - Available sections: {list(full_results.keys())}")
        
        if 'structured_report' in full_results:
            report = full_results['structured_report']
            builder_info = report['builder_info']
            summary = report['summary']
            
            print(f"   - Builder: {builder_info['builder_class']}")
            print(f"   - Step Type: {builder_info['sagemaker_step_type']}")
            print(f"   - Pass Rate: {summary['pass_rate']:.1f}%")
            if 'overall_score' in summary:
                print(f"   - Quality Score: {summary['overall_score']:.1f}/100")
        
        # Example 3: Convenience methods
        print("\n\n3️⃣  Using Convenience Methods:")
        print("-" * 40)
        
        tester_convenience = UniversalStepBuilderTest(TabularPreprocessingStepBuilder, verbose=False)
        
        # Legacy method (backward compatibility)
        legacy_results = tester_convenience.run_all_tests_legacy()
        print(f"✅ Legacy method: {len(legacy_results)} raw test results")
        
        # Scoring method
        scoring_results = tester_convenience.run_all_tests_with_scoring()
        print(f"✅ Scoring method: includes scoring data")
        
        # Full report method
        full_report_results = tester_convenience.run_all_tests_with_full_report()
        print(f"✅ Full report method: {len(full_report_results)} sections")
        
        # Example 4: Export to JSON
        print("\n\n4️⃣  Exporting Results to JSON:")
        print("-" * 40)
        
        output_path = "test_reports/example_builder_report.json"
        json_content = tester_convenience.export_results_to_json(output_path)
        
        print(f"✅ Results exported to: {output_path}")
        print(f"   - JSON size: {len(json_content)} characters")
        
        # Show a sample of the JSON structure
        try:
            data = json.loads(json_content)
            print(f"   - JSON sections: {list(data.keys())}")
            if 'scoring' in data and 'overall' in data['scoring']:
                overall = data['scoring']['overall']
                print(f"   - Overall score: {overall['score']:.1f}/100 ({overall['rating']})")
        except Exception as e:
            print(f"   - JSON parsing note: {e}")
        
        print("\n🎉 Demonstration completed successfully!")
        print("\nKey Benefits of the Enhanced System:")
        print("  ✅ Backward compatible - existing code still works")
        print("  ✅ Integrated scoring - quantitative quality assessment")
        print("  ✅ Structured reporting - consistent with alignment validation")
        print("  ✅ Multiple output formats - raw, scored, structured, JSON")
        print("  ✅ Flexible usage - enable/disable features as needed")
        
    except ImportError as e:
        print(f"❌ Could not import required modules: {e}")
        print("   This is expected if the step builders are not available in the current environment.")
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def show_usage_patterns():
    """Show different usage patterns for the enhanced system."""
    
    print("\n📚 Usage Patterns:")
    print("=" * 50)
    
    usage_examples = [
        {
            "title": "Basic Testing (Legacy Compatible)",
            "code": """
# This maintains backward compatibility
tester = UniversalStepBuilderTest(BuilderClass, enable_scoring=False)
results = tester.run_all_tests()  # Returns raw dict like before
""",
            "description": "Use when you want the original behavior"
        },
        {
            "title": "Enhanced Testing with Scoring",
            "code": """
# New default behavior with scoring
tester = UniversalStepBuilderTest(BuilderClass, verbose=True)
results = tester.run_all_tests()
score = results['scoring']['overall']['score']
""",
            "description": "Get quantitative quality scores alongside test results"
        },
        {
            "title": "Full Reporting for Analysis",
            "code": """
# Complete reporting with structured output
tester = UniversalStepBuilderTest(
    BuilderClass, 
    enable_scoring=True,
    enable_structured_reporting=True
)
results = tester.run_all_tests()
# Access: results['test_results'], results['scoring'], results['structured_report']
""",
            "description": "Generate comprehensive reports for detailed analysis"
        },
        {
            "title": "Convenient Methods",
            "code": """
tester = UniversalStepBuilderTest(BuilderClass)

# Different ways to run tests
raw_results = tester.run_all_tests_legacy()
scored_results = tester.run_all_tests_with_scoring()
full_results = tester.run_all_tests_with_full_report()

# Export to JSON
tester.export_results_to_json("report.json")
""",
            "description": "Use convenience methods for specific needs"
        }
    ]
    
    for i, example in enumerate(usage_examples, 1):
        print(f"\n{i}️⃣  {example['title']}:")
        print(f"   {example['description']}")
        print(f"   Code:")
        for line in example['code'].strip().split('\n'):
            print(f"     {line}")


if __name__ == "__main__":
    demonstrate_enhanced_testing()
    show_usage_patterns()
