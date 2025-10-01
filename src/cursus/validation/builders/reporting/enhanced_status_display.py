"""
Enhanced status display with icons from legacy scripts.

This module provides enhanced status formatting and summary display functionality
to match the visual output style of legacy report generation scripts.
"""

from typing import Dict, Any
from .step_type_color_scheme import StepTypeColorScheme


class EnhancedStatusDisplay:
    """Enhanced status display with icons from legacy scripts."""
    
    @staticmethod
    def format_builder_status(canonical_name: str, test_results: Dict[str, Any], 
                            step_type: str = None) -> str:
        """Format builder status with icon and pass rate."""
        try:
            # Calculate pass rate
            if 'test_results' in test_results:
                raw_results = test_results['test_results']
                total_tests = len(raw_results)
                passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            else:
                pass_rate = 0
            
            # Determine status and icon
            if pass_rate >= 80:
                status_icon = "✅"
                status_text = "PASSING"
            elif pass_rate >= 60:
                status_icon = "⚠️"
                status_text = "WARNING"
            else:
                status_icon = "❌"
                status_text = "FAILING"
            
            # Add step type if available
            type_info = f" [{step_type}]" if step_type else ""
            
            return f"{status_icon} {canonical_name}{type_info}: {status_text} ({pass_rate:.1f}%)"
            
        except Exception as e:
            return f"❓ {canonical_name}: ERROR ({e})"
    
    @staticmethod
    def print_builder_summary(results: Dict[str, Any], step_catalog) -> None:
        """Print comprehensive builder summary with enhanced formatting."""
        print(f"\n{'='*80}")
        print("📊 ENHANCED BUILDER TEST SUMMARY")
        print(f"{'='*80}")
        
        # Group by step type
        step_type_groups = {}
        for canonical_name, test_results in results.items():
            try:
                step_info = step_catalog.get_step_info(canonical_name)
                step_type = step_info.registry_data.get('sagemaker_step_type', 'Unknown') if step_info else 'Unknown'
                
                if step_type not in step_type_groups:
                    step_type_groups[step_type] = []
                
                status_line = EnhancedStatusDisplay.format_builder_status(
                    canonical_name, test_results, step_type
                )
                step_type_groups[step_type].append((canonical_name, status_line, test_results))
                
            except Exception as e:
                if 'Unknown' not in step_type_groups:
                    step_type_groups['Unknown'] = []
                step_type_groups['Unknown'].append((canonical_name, f"❓ {canonical_name}: ERROR ({e})", {}))
        
        # Print by step type with colors
        for step_type, builders in sorted(step_type_groups.items()):
            color = StepTypeColorScheme.get_color_for_step_type(step_type)
            print(f"\n🔧 {step_type} Steps ({len(builders)} builders):")
            print(f"   Color: {color}")
            
            for canonical_name, status_line, test_results in sorted(builders, key=lambda x: x[1]):
                print(f"   {status_line}")
        
        # Overall statistics
        total_builders = len(results)
        successful_builders = sum(1 for _, _, test_results in 
                                [item for sublist in step_type_groups.values() for item in sublist]
                                if EnhancedStatusDisplay._is_passing(test_results))
        
        success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Builders: {total_builders}")
        print(f"   Successful: {successful_builders} ({success_rate:.1f}%)")
        print(f"   Failed: {total_builders - successful_builders}")
        print(f"   Step Types: {len(step_type_groups)}")
        print(f"{'='*80}")
    
    @staticmethod
    def _is_passing(test_results: Dict[str, Any]) -> bool:
        """Check if test results indicate passing status."""
        try:
            if 'test_results' in test_results:
                raw_results = test_results['test_results']
                total_tests = len(raw_results)
                passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                return pass_rate >= 60
            return False
        except Exception:
            return False
    
    @staticmethod
    def get_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics for test results."""
        total_builders = len(results)
        successful_builders = sum(1 for test_results in results.values() 
                                if EnhancedStatusDisplay._is_passing(test_results))
        
        success_rate = (successful_builders / total_builders * 100) if total_builders > 0 else 0
        
        return {
            'total_builders': total_builders,
            'successful_builders': successful_builders,
            'failed_builders': total_builders - successful_builders,
            'success_rate': success_rate
        }
