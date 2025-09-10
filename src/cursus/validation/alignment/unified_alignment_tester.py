"""
Unified Alignment Tester - Main orchestrator for all alignment validation levels.

Coordinates validation across all four alignment levels:
1. Script ↔ Contract Alignment
2. Contract ↔ Specification Alignment  
3. Specification ↔ Dependencies Alignment
4. Builder ↔ Configuration Alignment
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .alignment_reporter import AlignmentReport, ValidationResult
from .alignment_utils import (
    SeverityLevel, AlignmentLevel, create_alignment_issue,
    detect_step_type_from_registry, detect_framework_from_imports,
    StepTypeAwareAlignmentIssue, create_step_type_aware_alignment_issue
)
from .step_type_enhancement_router import StepTypeEnhancementRouter
from .script_contract_alignment import ScriptContractAlignmentTester
from .contract_spec_alignment import ContractSpecificationAlignmentTester
from .spec_dependency_alignment import SpecificationDependencyAlignmentTester
from .level3_validation_config import Level3ValidationConfig, ValidationMode
from .builder_config_alignment import BuilderConfigurationAlignmentTester


class UnifiedAlignmentTester:
    """
    Main orchestrator for comprehensive alignment validation.
    
    Coordinates all four levels of alignment testing and produces
    a unified report with actionable recommendations.
    """
    
    def __init__(self, 
                 scripts_dir: str = "src/cursus/steps/scripts",
                 contracts_dir: str = "src/cursus/steps/contracts",
                 specs_dir: str = "src/cursus/steps/specs",
                 builders_dir: str = "src/cursus/steps/builders",
                 configs_dir: str = "src/cursus/steps/configs",
                 level3_validation_mode: str = "relaxed"):
        """
        Initialize the unified alignment tester.
        
        Args:
            scripts_dir: Directory containing processing scripts
            contracts_dir: Directory containing script contracts
            specs_dir: Directory containing step specifications
            builders_dir: Directory containing step builders
            configs_dir: Directory containing step configurations
            level3_validation_mode: Level 3 validation mode ('strict', 'relaxed', 'permissive')
        """
        self.scripts_dir = Path(scripts_dir).resolve()
        self.contracts_dir = Path(contracts_dir).resolve()
        self.specs_dir = Path(specs_dir).resolve()
        self.builders_dir = Path(builders_dir).resolve()
        self.configs_dir = Path(configs_dir).resolve()
        
        # Configure Level 3 validation based on mode
        if level3_validation_mode == "strict":
            level3_config = Level3ValidationConfig.create_strict_config()
        elif level3_validation_mode == "relaxed":
            level3_config = Level3ValidationConfig.create_relaxed_config()
        elif level3_validation_mode == "permissive":
            level3_config = Level3ValidationConfig.create_permissive_config()
        else:
            level3_config = Level3ValidationConfig.create_relaxed_config()  # Default
            print(f"⚠️  Unknown Level 3 validation mode '{level3_validation_mode}', using 'relaxed' mode")
        
        # Initialize level-specific testers
        self.level1_tester = ScriptContractAlignmentTester(scripts_dir, contracts_dir, builders_dir)
        self.level2_tester = ContractSpecificationAlignmentTester(contracts_dir, specs_dir)
        self.level3_tester = SpecificationDependencyAlignmentTester(specs_dir, level3_config)
        self.level4_tester = BuilderConfigurationAlignmentTester(str(self.builders_dir), str(self.configs_dir))
        
        self.report = AlignmentReport()
        
        # Store configuration for reporting
        self.level3_config = level3_config
        
        # Phase 1 Enhancement: Step type awareness feature flag
        self.enable_step_type_awareness = os.getenv('ENABLE_STEP_TYPE_AWARENESS', 'true').lower() == 'true'
        
        # Phase 3 Enhancement: Step Type Enhancement System
        self.step_type_enhancement_router = StepTypeEnhancementRouter()
    
    def run_full_validation(self, 
                           target_scripts: Optional[List[str]] = None,
                           skip_levels: Optional[List[int]] = None) -> AlignmentReport:
        """
        Run comprehensive alignment validation across all levels.
        
        Args:
            target_scripts: Specific scripts to validate (None for all)
            skip_levels: Alignment levels to skip (1-4)
            
        Returns:
            Comprehensive alignment report
        """
        skip_levels = skip_levels or []
        
        print("🔍 Starting Unified Alignment Validation...")
        
        # Level 1: Script ↔ Contract Alignment
        if 1 not in skip_levels:
            print("\n📝 Level 1: Validating Script ↔ Contract Alignment...")
            try:
                self._run_level1_validation(target_scripts)
            except Exception as e:
                print(f"⚠️  Level 1 validation encountered an error: {e}")
        
        # Level 2: Contract ↔ Specification Alignment
        if 2 not in skip_levels:
            print("\n📋 Level 2: Validating Contract ↔ Specification Alignment...")
            try:
                self._run_level2_validation(target_scripts)
            except Exception as e:
                print(f"⚠️  Level 2 validation encountered an error: {e}")
        
        # Level 3: Specification ↔ Dependencies Alignment
        if 3 not in skip_levels:
            print("\n🔗 Level 3: Validating Specification ↔ Dependencies Alignment...")
            try:
                self._run_level3_validation(target_scripts)
            except Exception as e:
                print(f"⚠️  Level 3 validation encountered an error: {e}")
        
        # Level 4: Builder ↔ Configuration Alignment
        if 4 not in skip_levels:
            print("\n⚙️  Level 4: Validating Builder ↔ Configuration Alignment...")
            try:
                self._run_level4_validation(target_scripts)
            except Exception as e:
                print(f"⚠️  Level 4 validation encountered an error: {e}")
        
        # Generate summary and recommendations
        print("\n📊 Generating alignment report...")
        self.report.generate_summary()
        
        # Print alignment scoring summary
        print("\n📈 Alignment Quality Scoring:")
        try:
            scorer = self.report.get_scorer()
            overall_score = scorer.calculate_overall_score()
            overall_rating = scorer.get_rating(overall_score)
            print(f"   Overall Score: {overall_score:.1f}/100 ({overall_rating})")
            
            # Print level scores
            level_names = {
                "level1_script_contract": "L1 Script↔Contract",
                "level2_contract_spec": "L2 Contract↔Spec",
                "level3_spec_dependencies": "L3 Spec↔Dependencies", 
                "level4_builder_config": "L4 Builder↔Config"
            }
            
            for level_key, level_name in level_names.items():
                score, passed, total = scorer.calculate_level_score(level_key)
                print(f"   {level_name}: {score:.1f}/100 ({passed}/{total} tests)")
                
        except Exception as e:
            print(f"   ⚠️  Scoring calculation failed: {e}")
        
        return self.report
    
    def run_level_validation(self, level: int, 
                           target_scripts: Optional[List[str]] = None) -> AlignmentReport:
        """
        Run validation for a specific alignment level.
        
        Args:
            level: Alignment level to validate (1-4)
            target_scripts: Specific scripts to validate
            
        Returns:
            Alignment report for the specified level
        """
        print(f"🔍 Running Level {level} Alignment Validation...")
        
        if level == 1:
            self._run_level1_validation(target_scripts)
        elif level == 2:
            self._run_level2_validation(target_scripts)
        elif level == 3:
            self._run_level3_validation(target_scripts)
        elif level == 4:
            self._run_level4_validation(target_scripts)
        else:
            raise ValueError(f"Invalid alignment level: {level}. Must be 1-4.")
        
        self.report.generate_summary()
        return self.report
    
    def _run_level1_validation(self, target_scripts: Optional[List[str]] = None):
        """Run Level 1: Script ↔ Contract alignment validation."""
        try:
            results = self.level1_tester.validate_all_scripts(target_scripts)
            
            for script_name, result in results.items():
                validation_result = ValidationResult(
                    test_name=f"script_contract_{script_name}",
                    passed=result.get('passed', False),
                    details=result
                )
                
                # Convert issues to AlignmentIssue objects
                for issue in result.get('issues', []):
                    alignment_issue = create_alignment_issue(
                        level=SeverityLevel(issue.get('severity', 'ERROR')),
                        category=issue.get('category', 'script_contract'),
                        message=issue.get('message', ''),
                        details=issue.get('details', {}),
                        recommendation=issue.get('recommendation'),
                        alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                    )
                    validation_result.add_issue(alignment_issue)
                
                # Phase 1 Enhancement: Add step type context to issues if enabled
                if self.enable_step_type_awareness:
                    self._add_step_type_context_to_issues(script_name, validation_result)
                
                # Phase 3 Enhancement: Apply step type-specific validation enhancements
                enhanced_result = self.step_type_enhancement_router.enhance_validation_results(
                    validation_result.details, script_name
                )
                
                # Merge enhanced issues into validation result
                if 'enhanced_issues' in enhanced_result:
                    for enhanced_issue_data in enhanced_result['enhanced_issues']:
                        enhanced_issue = create_alignment_issue(
                            level=SeverityLevel(enhanced_issue_data.get('severity', 'INFO')),
                            category=enhanced_issue_data.get('category', 'step_type_enhancement'),
                            message=enhanced_issue_data.get('message', ''),
                            details=enhanced_issue_data.get('details', {}),
                            recommendation=enhanced_issue_data.get('recommendation'),
                            alignment_level=AlignmentLevel.SCRIPT_CONTRACT
                        )
                        validation_result.add_issue(enhanced_issue)
                
                self.report.add_level1_result(script_name, validation_result)
                
        except Exception as e:
            # Create error result for level 1
            error_result = ValidationResult(
                test_name="level1_validation",
                passed=False,
                details={'error': str(e)}
            )
            
            error_issue = create_alignment_issue(
                level=SeverityLevel.CRITICAL,
                category='validation_error',
                message=f"Level 1 validation failed: {str(e)}",
                alignment_level=AlignmentLevel.SCRIPT_CONTRACT
            )
            error_result.add_issue(error_issue)
            
            self.report.add_level1_result("validation_error", error_result)
    
    def _run_level2_validation(self, target_scripts: Optional[List[str]] = None):
        """Run Level 2: Contract ↔ Specification alignment validation."""
        try:
            results = self.level2_tester.validate_all_contracts(target_scripts)
            
            for contract_name, result in results.items():
                validation_result = ValidationResult(
                    test_name=f"contract_spec_{contract_name}",
                    passed=result.get('passed', False),
                    details=result
                )
                
                # Convert issues to AlignmentIssue objects
                for issue in result.get('issues', []):
                    alignment_issue = create_alignment_issue(
                        level=SeverityLevel(issue.get('severity', 'ERROR')),
                        category=issue.get('category', 'contract_specification'),
                        message=issue.get('message', ''),
                        details=issue.get('details', {}),
                        recommendation=issue.get('recommendation'),
                        alignment_level=AlignmentLevel.CONTRACT_SPECIFICATION
                    )
                    validation_result.add_issue(alignment_issue)
                
                self.report.add_level2_result(contract_name, validation_result)
                
        except Exception as e:
            # Create error result for level 2
            error_result = ValidationResult(
                test_name="level2_validation",
                passed=False,
                details={'error': str(e)}
            )
            
            error_issue = create_alignment_issue(
                level=SeverityLevel.CRITICAL,
                category='validation_error',
                message=f"Level 2 validation failed: {str(e)}",
                alignment_level=AlignmentLevel.CONTRACT_SPECIFICATION
            )
            error_result.add_issue(error_issue)
            
            self.report.add_level2_result("validation_error", error_result)
    
    def _run_level3_validation(self, target_scripts: Optional[List[str]] = None):
        """Run Level 3: Specification ↔ Dependencies alignment validation."""
        try:
            results = self.level3_tester.validate_all_specifications(target_scripts)
            
            for spec_name, result in results.items():
                validation_result = ValidationResult(
                    test_name=f"spec_dependency_{spec_name}",
                    passed=result.get('passed', False),
                    details=result
                )
                
                # Convert issues to AlignmentIssue objects
                for issue in result.get('issues', []):
                    alignment_issue = create_alignment_issue(
                        level=SeverityLevel(issue.get('severity', 'ERROR')),
                        category=issue.get('category', 'specification_dependency'),
                        message=issue.get('message', ''),
                        details=issue.get('details', {}),
                        recommendation=issue.get('recommendation'),
                        alignment_level=AlignmentLevel.SPECIFICATION_DEPENDENCY
                    )
                    validation_result.add_issue(alignment_issue)
                
                self.report.add_level3_result(spec_name, validation_result)
                
        except Exception as e:
            # Create error result for level 3
            error_result = ValidationResult(
                test_name="level3_validation",
                passed=False,
                details={'error': str(e)}
            )
            
            error_issue = create_alignment_issue(
                level=SeverityLevel.CRITICAL,
                category='validation_error',
                message=f"Level 3 validation failed: {str(e)}",
                alignment_level=AlignmentLevel.SPECIFICATION_DEPENDENCY
            )
            error_result.add_issue(error_issue)
            
            self.report.add_level3_result("validation_error", error_result)
    
    def _run_level4_validation(self, target_scripts: Optional[List[str]] = None):
        """Run Level 4: Builder ↔ Configuration alignment validation."""
        try:
            results = self.level4_tester.validate_all_builders(target_scripts)
            
            for builder_name, result in results.items():
                validation_result = ValidationResult(
                    test_name=f"builder_config_{builder_name}",
                    passed=result.get('passed', False),
                    details=result
                )
                
                # Convert issues to AlignmentIssue objects
                for issue in result.get('issues', []):
                    alignment_issue = create_alignment_issue(
                        level=SeverityLevel(issue.get('severity', 'ERROR')),
                        category=issue.get('category', 'builder_configuration'),
                        message=issue.get('message', ''),
                        details=issue.get('details', {}),
                        recommendation=issue.get('recommendation'),
                        alignment_level=AlignmentLevel.BUILDER_CONFIGURATION
                    )
                    validation_result.add_issue(alignment_issue)
                
                self.report.add_level4_result(builder_name, validation_result)
                
        except Exception as e:
            # Create error result for level 4
            error_result = ValidationResult(
                test_name="level4_validation",
                passed=False,
                details={'error': str(e)}
            )
            
            error_issue = create_alignment_issue(
                level=SeverityLevel.CRITICAL,
                category='validation_error',
                message=f"Level 4 validation failed: {str(e)}",
                alignment_level=AlignmentLevel.BUILDER_CONFIGURATION
            )
            error_result.add_issue(error_issue)
            
            self.report.add_level4_result("validation_error", error_result)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a high-level summary of validation results."""
        if not self.report.summary:
            self.report.generate_summary()
        
        # Get scoring information
        scoring_info = {}
        try:
            scorer = self.report.get_scorer()
            overall_score = scorer.calculate_overall_score()
            overall_rating = scorer.get_rating(overall_score)
            
            scoring_info = {
                'overall_score': overall_score,
                'quality_rating': overall_rating,
                'level_scores': {}
            }
            
            # Add level scores
            level_keys = ["level1_script_contract", "level2_contract_spec", 
                         "level3_spec_dependencies", "level4_builder_config"]
            for level_key in level_keys:
                score, passed, total = scorer.calculate_level_score(level_key)
                scoring_info['level_scores'][level_key] = score
                
        except Exception as e:
            # If scoring fails, provide default values
            scoring_info = {
                'overall_score': 0.0,
                'quality_rating': 'Unknown',
                'level_scores': {
                    'level1_script_contract': 0.0,
                    'level2_contract_spec': 0.0,
                    'level3_spec_dependencies': 0.0,
                    'level4_builder_config': 0.0
                }
            }
        
        return {
            'overall_status': 'PASSING' if self.report.is_passing() else 'FAILING',
            'total_tests': self.report.summary.total_tests,
            'pass_rate': self.report.summary.pass_rate,
            'critical_issues': self.report.summary.critical_issues,
            'error_issues': self.report.summary.error_issues,
            'warning_issues': self.report.summary.warning_issues,
            'level_breakdown': {
                'level1_tests': len(self.report.level1_results),
                'level2_tests': len(self.report.level2_results),
                'level3_tests': len(self.report.level3_results),
                'level4_tests': len(self.report.level4_results)
            },
            'recommendations_count': len(self.report.get_recommendations()),
            'scoring': scoring_info
        }
    
    def export_report(self, format: str = 'json', output_path: Optional[str] = None, 
                     generate_chart: bool = True, script_name: str = "alignment_validation") -> str:
        """
        Export the alignment report in the specified format with optional visualization.
        
        Args:
            format: Export format ('json' or 'html')
            output_path: Optional path to save the report
            generate_chart: Whether to generate alignment score chart
            script_name: Name for the chart file
            
        Returns:
            Report content as string
        """
        if format.lower() == 'json':
            content = self.report.export_to_json()
        elif format.lower() == 'html':
            content = self.report.export_to_html()
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 Report exported to: {output_path}")
        
        # Generate alignment score chart if requested (regardless of output_path)
        if generate_chart:
            try:
                output_dir = str(Path(output_path).parent) if output_path else "."
                chart_path = self.report.get_scorer().generate_chart(script_name, output_dir)
                if chart_path:
                    print(f"📊 Alignment score chart generated: {chart_path}")
                else:
                    print("⚠️  Chart generation skipped (matplotlib not available)")
            except Exception as e:
                print(f"⚠️  Chart generation failed: {e}")
        
        return content
    
    def print_summary(self):
        """Print a formatted summary of validation results."""
        self.report.print_summary()
    
    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """Get all critical issues that require immediate attention."""
        critical_issues = []
        
        for issue in self.report.get_critical_issues():
            critical_issues.append({
                'level': issue.level.value,
                'category': issue.category,
                'message': issue.message,
                'details': issue.details,
                'recommendation': issue.recommendation,
                'alignment_level': issue.alignment_level.value if issue.alignment_level else None
            })
        
        return critical_issues
    
    def validate_specific_script(self, script_name: str) -> Dict[str, Any]:
        """
        Run comprehensive validation for a specific script across all levels.
        
        Args:
            script_name: Name of the script to validate
            
        Returns:
            Validation results for the specific script
        """
        print(f"🔍 Validating script: {script_name}")
        
        results = {
            'script_name': script_name,
            'level1': {},
            'level2': {},
            'level3': {},
            'level4': {},
            'overall_status': 'UNKNOWN'
        }
        
        # Run validation for each level
        try:
            # Level 1: Script ↔ Contract
            level1_results = self.level1_tester.validate_script(script_name)
            results['level1'] = level1_results
            
            # Level 2: Contract ↔ Specification
            level2_results = self.level2_tester.validate_contract(script_name)
            results['level2'] = level2_results
            
            # Level 3: Specification ↔ Dependencies
            level3_results = self.level3_tester.validate_specification(script_name)
            results['level3'] = level3_results
            
            # Level 4: Builder ↔ Configuration
            level4_results = self.level4_tester.validate_builder(script_name)
            results['level4'] = level4_results
            
            # Determine overall status
            all_passed = all([
                results['level1'].get('passed', False),
                results['level2'].get('passed', False),
                results['level3'].get('passed', False),
                results['level4'].get('passed', False)
            ])
            
            results['overall_status'] = 'PASSING' if all_passed else 'FAILING'
            
            # Add scoring information
            try:
                # Create temporary validation results to calculate scoring
                temp_report = AlignmentReport()
                
                # Add results to temporary report for scoring calculation
                for level_name, level_result in [
                    ('level1', level1_results), ('level2', level2_results),
                    ('level3', level3_results), ('level4', level4_results)
                ]:
                    if level_result.get('passed') is not None:
                        validation_result = ValidationResult(
                            test_name=f"{level_name}_{script_name}",
                            passed=level_result.get('passed', False),
                            details=level_result
                        )
                        
                        if level_name == 'level1':
                            temp_report.add_level1_result(script_name, validation_result)
                        elif level_name == 'level2':
                            temp_report.add_level2_result(script_name, validation_result)
                        elif level_name == 'level3':
                            temp_report.add_level3_result(script_name, validation_result)
                        elif level_name == 'level4':
                            temp_report.add_level4_result(script_name, validation_result)
                
                temp_report.generate_summary()
                scorer = temp_report.get_scorer()
                overall_score = scorer.calculate_overall_score()
                overall_rating = scorer.get_rating(overall_score)
                
                scoring_info = {
                    'overall_score': overall_score,
                    'quality_rating': overall_rating,
                    'level_scores': {}
                }
                
                # Add level scores
                level_keys = ["level1_script_contract", "level2_contract_spec", 
                             "level3_spec_dependencies", "level4_builder_config"]
                for level_key in level_keys:
                    score, passed, total = scorer.calculate_level_score(level_key)
                    scoring_info['level_scores'][level_key] = score
                
                results['scoring'] = scoring_info
                
            except Exception as e:
                # If scoring fails, provide default values
                results['scoring'] = {
                    'overall_score': 0.0,
                    'quality_rating': 'Unknown',
                    'level_scores': {
                        'level1_script_contract': 0.0,
                        'level2_contract_spec': 0.0,
                        'level3_spec_dependencies': 0.0,
                        'level4_builder_config': 0.0
                    }
                }
            
        except Exception as e:
            results['error'] = str(e)
            results['overall_status'] = 'ERROR'
        
        return results
    
    def discover_scripts(self) -> List[str]:
        """Discover all Python scripts in the scripts directory."""
        scripts = []
        
        if self.scripts_dir.exists():
            for script_file in self.scripts_dir.glob("*.py"):
                if not script_file.name.startswith('__'):
                    scripts.append(script_file.stem)
        
        return sorted(scripts)
    
    def get_alignment_status_matrix(self) -> Dict[str, Dict[str, str]]:
        """
        Get a matrix showing alignment status for each script across all levels.
        
        Returns:
            Matrix with script names as keys and level statuses as values
        """
        matrix = {}
        scripts = self.discover_scripts()
        
        for script in scripts:
            matrix[script] = {
                'level1': 'UNKNOWN',
                'level2': 'UNKNOWN', 
                'level3': 'UNKNOWN',
                'level4': 'UNKNOWN'
            }
        
        # Update with actual results if available
        for script_name, result in self.report.level1_results.items():
            if script_name in matrix:
                matrix[script_name]['level1'] = 'PASSING' if result.passed else 'FAILING'
        
        for contract_name, result in self.report.level2_results.items():
            if contract_name in matrix:
                matrix[contract_name]['level2'] = 'PASSING' if result.passed else 'FAILING'
        
        for spec_name, result in self.report.level3_results.items():
            if spec_name in matrix:
                matrix[spec_name]['level3'] = 'PASSING' if result.passed else 'FAILING'
        
        for builder_name, result in self.report.level4_results.items():
            if builder_name in matrix:
                matrix[builder_name]['level4'] = 'PASSING' if result.passed else 'FAILING'
        
        return matrix
    
    def _add_step_type_context_to_issues(self, script_name: str, validation_result: ValidationResult):
        """
        Phase 1 Enhancement: Add step type context to validation issues.
        
        Args:
            script_name: Name of the script being validated
            validation_result: ValidationResult to enhance with step type context
        """
        try:
            # Detect step type from registry
            step_type = detect_step_type_from_registry(script_name)
            
            # Detect framework if possible (requires script analysis)
            framework = None
            try:
                # Try to get framework from script analysis if available
                script_path = self.scripts_dir / f"{script_name}.py"
                if script_path.exists():
                    from .framework_patterns import detect_framework_from_script_content
                    with open(script_path, 'r', encoding='utf-8') as f:
                        script_content = f.read()
                    framework = detect_framework_from_script_content(script_content)
            except Exception as e:
                # Framework detection is optional, continue without it
                print(f"⚠️  Framework detection failed for {script_name}: {e}")
            
            # Add step type context to existing issues
            for issue in validation_result.issues:
                if hasattr(issue, 'step_type'):
                    # Already a StepTypeAwareAlignmentIssue, update it
                    issue.step_type = step_type
                    if framework:
                        issue.framework_context = framework
                else:
                    # Convert to step type-aware issue
                    step_type_issue = create_step_type_aware_alignment_issue(
                        level=issue.level,
                        category=issue.category,
                        message=issue.message,
                        step_type=step_type,
                        framework_context=framework,
                        details=issue.details,
                        recommendation=issue.recommendation,
                        alignment_level=issue.alignment_level
                    )
                    # Replace the original issue
                    validation_result.issues[validation_result.issues.index(issue)] = step_type_issue
            
            # Add step type information to validation result details
            validation_result.details['step_type'] = step_type
            if framework:
                validation_result.details['framework'] = framework
                
        except Exception as e:
            # Step type enhancement is optional, don't fail validation if it fails
            print(f"⚠️  Step type enhancement failed for {script_name}: {e}")
