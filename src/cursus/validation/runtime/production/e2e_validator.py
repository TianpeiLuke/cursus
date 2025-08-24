"""
End-to-End Validation Framework for Pipeline Runtime Testing.

This module provides comprehensive validation of pipeline configurations
with real data and production-like scenarios.
"""

import os
import time
import json
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator
import yaml

from ..core.pipeline_script_executor import PipelineScriptExecutor
from ..execution.pipeline_executor import PipelineExecutor
from ..integration.real_data_tester import RealDataTester
from ..utils.execution_context import ExecutionContext

logger = logging.getLogger(__name__)


class E2ETestScenario(BaseModel):
    """Configuration for an end-to-end test scenario."""
    
    scenario_name: str = Field(..., description="Unique name for the test scenario")
    pipeline_config_path: str = Field(..., description="Path to pipeline configuration file")
    expected_steps: List[str] = Field(..., description="Expected pipeline steps to be executed")
    data_source: str = Field(default="synthetic", description="Data source: synthetic, real, or custom")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Custom validation rules")
    timeout_minutes: int = Field(default=30, description="Maximum execution time in minutes")
    memory_limit_gb: float = Field(default=4.0, description="Memory limit in GB")
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing scenarios")
    
    @field_validator('timeout_minutes')
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0 or v > 180:
            raise ValueError("Timeout must be between 1 and 180 minutes")
        return v
    
    @field_validator('memory_limit_gb')
    @classmethod
    def validate_memory_limit(cls, v):
        if v <= 0 or v > 32:
            raise ValueError("Memory limit must be between 0 and 32 GB")
        return v


class E2ETestResult(BaseModel):
    """Result of an end-to-end test scenario execution."""
    
    scenario_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    total_duration: float = Field(..., description="Total execution time in seconds")
    peak_memory_usage: float = Field(..., description="Peak memory usage in MB")
    steps_executed: int
    steps_failed: int
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    error_details: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of executed steps."""
        if self.steps_executed == 0:
            return 0.0
        return (self.steps_executed - self.steps_failed) / self.steps_executed


class E2EValidationReport(BaseModel):
    """Comprehensive report of end-to-end validation results."""
    
    report_id: str
    generation_time: datetime
    total_scenarios: int
    successful_scenarios: int
    failed_scenarios: int
    total_execution_time: float
    average_execution_time: float
    peak_memory_usage: float
    scenario_results: List[E2ETestResult]
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_scenarios == 0:
            return 0.0
        return self.successful_scenarios / self.total_scenarios


class EndToEndValidator:
    """
    End-to-End validation framework for pipeline runtime testing.
    
    Provides comprehensive testing capabilities with real pipeline configurations,
    performance monitoring, and detailed reporting.
    """
    
    def __init__(self, workspace_dir: str = None, config: Dict[str, Any] = None):
        """
        Initialize the E2E validator.
        
        Args:
            workspace_dir: Directory for test workspace and outputs
            config: Configuration dictionary for validator settings
        """
        self.workspace_dir = Path(workspace_dir or "./e2e_test_workspace")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        
        # Initialize components
        self.pipeline_executor = PipelineExecutor()
        self.real_data_tester = RealDataTester()
        
        # Performance monitoring
        self.process = psutil.Process()
        
        logger.info(f"E2E Validator initialized with workspace: {self.workspace_dir}")
    
    def discover_test_scenarios(self, scenarios_dir: str) -> List[E2ETestScenario]:
        """
        Discover test scenarios from a directory.
        
        Args:
            scenarios_dir: Directory containing scenario configuration files
            
        Returns:
            List of discovered test scenarios
        """
        scenarios = []
        scenarios_path = Path(scenarios_dir)
        
        if not scenarios_path.exists():
            logger.warning(f"Scenarios directory not found: {scenarios_dir}")
            return scenarios
        
        # Look for YAML and JSON scenario files
        for pattern in ["*.yaml", "*.yml", "*.json"]:
            for scenario_file in scenarios_path.glob(pattern):
                try:
                    scenario = self._load_scenario_from_file(scenario_file)
                    scenarios.append(scenario)
                    logger.info(f"Discovered scenario: {scenario.scenario_name}")
                except Exception as e:
                    logger.error(f"Failed to load scenario from {scenario_file}: {e}")
        
        logger.info(f"Discovered {len(scenarios)} test scenarios")
        return scenarios
    
    def _load_scenario_from_file(self, scenario_file: Path) -> E2ETestScenario:
        """Load a test scenario from a configuration file."""
        with open(scenario_file, 'r') as f:
            if scenario_file.suffix.lower() == '.json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        # Add file path if not specified
        if 'scenario_name' not in data:
            data['scenario_name'] = scenario_file.stem
        
        return E2ETestScenario(**data)
    
    def execute_scenario(self, scenario: E2ETestScenario) -> E2ETestResult:
        """
        Execute a single end-to-end test scenario.
        
        Args:
            scenario: Test scenario configuration
            
        Returns:
            Test execution result
        """
        logger.info(f"Executing E2E scenario: {scenario.scenario_name}")
        
        start_time = datetime.now()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        result = E2ETestResult(
            scenario_name=scenario.scenario_name,
            success=False,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            total_duration=0.0,
            peak_memory_usage=initial_memory,
            steps_executed=0,
            steps_failed=0
        )
        
        try:
            # Set up execution context
            context = ExecutionContext(
                workspace_dir=str(self.workspace_dir / scenario.scenario_name),
                environment_variables=scenario.environment_variables,
                timeout_seconds=scenario.timeout_minutes * 60,
                memory_limit_mb=scenario.memory_limit_gb * 1024
            )
            
            # Monitor memory usage during execution
            memory_monitor = self._start_memory_monitoring()
            
            # Execute the pipeline
            execution_result = self._execute_pipeline_scenario(scenario, context)
            
            # Stop memory monitoring
            peak_memory = self._stop_memory_monitoring(memory_monitor)
            
            # Update result with execution details
            result.steps_executed = execution_result.get('steps_executed', 0)
            result.steps_failed = execution_result.get('steps_failed', 0)
            result.success = execution_result.get('success', False)
            result.validation_results = execution_result.get('validation_results', {})
            result.performance_metrics = execution_result.get('performance_metrics', {})
            result.warnings = execution_result.get('warnings', [])
            
            # Validate expected steps
            self._validate_expected_steps(scenario, execution_result, result)
            
        except Exception as e:
            result.success = False
            result.error_details = str(e)
            logger.error(f"Scenario execution failed: {e}")
        
        finally:
            end_time = datetime.now()
            result.end_time = end_time
            result.total_duration = (end_time - start_time).total_seconds()
            result.peak_memory_usage = peak_memory
            
            # Add resource usage summary
            result.resource_usage = {
                'memory_increase_mb': peak_memory - initial_memory,
                'execution_time_seconds': result.total_duration,
                'memory_efficiency': peak_memory / (scenario.memory_limit_gb * 1024)
            }
        
        logger.info(f"Scenario {scenario.scenario_name} completed: "
                   f"Success={result.success}, Duration={result.total_duration:.2f}s")
        
        return result
    
    def _execute_pipeline_scenario(self, scenario: E2ETestScenario, context: ExecutionContext) -> Dict[str, Any]:
        """Execute the actual pipeline scenario."""
        try:
            # Load pipeline configuration
            config_path = Path(scenario.pipeline_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Pipeline config not found: {scenario.pipeline_config_path}")
            
            # Execute pipeline based on data source
            if scenario.data_source == "real":
                execution_result = self.real_data_tester.test_with_real_data(
                    str(config_path),
                    context.workspace_dir
                )
            else:
                execution_result = self.pipeline_executor.execute_pipeline(
                    str(config_path),
                    context.workspace_dir,
                    use_synthetic_data=(scenario.data_source == "synthetic")
                )
            
            return execution_result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'steps_executed': 0,
                'steps_failed': 1
            }
    
    def _validate_expected_steps(self, scenario: E2ETestScenario, execution_result: Dict[str, Any], result: E2ETestResult):
        """Validate that expected steps were executed."""
        executed_steps = execution_result.get('executed_steps', [])
        
        for expected_step in scenario.expected_steps:
            if expected_step not in executed_steps:
                result.warnings.append(f"Expected step not executed: {expected_step}")
        
        # Apply custom validation rules
        for rule_name, rule_config in scenario.validation_rules.items():
            try:
                self._apply_validation_rule(rule_name, rule_config, execution_result, result)
            except Exception as e:
                result.warnings.append(f"Validation rule '{rule_name}' failed: {e}")
    
    def _apply_validation_rule(self, rule_name: str, rule_config: Any, execution_result: Dict[str, Any], result: E2ETestResult):
        """Apply a custom validation rule."""
        if rule_name == "min_execution_time":
            if result.total_duration < rule_config:
                result.warnings.append(f"Execution too fast: {result.total_duration}s < {rule_config}s")
        
        elif rule_name == "max_memory_usage":
            if result.peak_memory_usage > rule_config:
                result.warnings.append(f"Memory usage too high: {result.peak_memory_usage}MB > {rule_config}MB")
        
        elif rule_name == "required_outputs":
            for output_path in rule_config:
                if not Path(output_path).exists():
                    result.warnings.append(f"Required output missing: {output_path}")
    
    def _start_memory_monitoring(self) -> Dict[str, Any]:
        """Start monitoring memory usage."""
        return {
            'start_time': time.time(),
            'initial_memory': self.process.memory_info().rss / 1024 / 1024,
            'peak_memory': self.process.memory_info().rss / 1024 / 1024
        }
    
    def _stop_memory_monitoring(self, monitor: Dict[str, Any]) -> float:
        """Stop monitoring and return peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        return max(monitor['peak_memory'], current_memory)
    
    def run_comprehensive_validation(self, scenarios_dir: str) -> E2EValidationReport:
        """
        Run comprehensive validation with all discovered scenarios.
        
        Args:
            scenarios_dir: Directory containing test scenarios
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Starting comprehensive E2E validation")
        
        # Discover scenarios
        scenarios = self.discover_test_scenarios(scenarios_dir)
        
        if not scenarios:
            logger.warning("No test scenarios found")
            return E2EValidationReport(
                report_id=f"e2e_report_{int(time.time())}",
                generation_time=datetime.now(),
                total_scenarios=0,
                successful_scenarios=0,
                failed_scenarios=0,
                total_execution_time=0.0,
                average_execution_time=0.0,
                peak_memory_usage=0.0,
                scenario_results=[]
            )
        
        # Execute all scenarios
        results = []
        total_execution_time = 0.0
        peak_memory = 0.0
        successful_count = 0
        
        for scenario in scenarios:
            result = self.execute_scenario(scenario)
            results.append(result)
            
            total_execution_time += result.total_duration
            peak_memory = max(peak_memory, result.peak_memory_usage)
            
            if result.success:
                successful_count += 1
        
        # Generate report
        report = E2EValidationReport(
            report_id=f"e2e_report_{int(time.time())}",
            generation_time=datetime.now(),
            total_scenarios=len(scenarios),
            successful_scenarios=successful_count,
            failed_scenarios=len(scenarios) - successful_count,
            total_execution_time=total_execution_time,
            average_execution_time=total_execution_time / len(scenarios) if scenarios else 0.0,
            peak_memory_usage=peak_memory,
            scenario_results=results
        )
        
        # Add summary metrics
        report.summary_metrics = self._generate_summary_metrics(results)
        report.recommendations = self._generate_recommendations(results)
        
        # Save report
        self._save_validation_report(report)
        
        logger.info(f"E2E validation completed: {successful_count}/{len(scenarios)} scenarios successful")
        
        return report
    
    def _generate_summary_metrics(self, results: List[E2ETestResult]) -> Dict[str, Any]:
        """Generate summary metrics from test results."""
        if not results:
            return {}
        
        return {
            'average_success_rate': sum(r.success_rate for r in results) / len(results),
            'average_memory_usage': sum(r.peak_memory_usage for r in results) / len(results),
            'total_warnings': sum(len(r.warnings) for r in results),
            'performance_distribution': {
                'fast_scenarios': len([r for r in results if r.total_duration < 30]),
                'medium_scenarios': len([r for r in results if 30 <= r.total_duration < 120]),
                'slow_scenarios': len([r for r in results if r.total_duration >= 120])
            }
        }
    
    def _generate_recommendations(self, results: List[E2ETestResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        slow_scenarios = [r for r in results if r.total_duration > 120]
        if slow_scenarios:
            recommendations.append(f"Consider optimizing {len(slow_scenarios)} slow scenarios (>2 minutes)")
        
        # Memory recommendations
        high_memory_scenarios = [r for r in results if r.peak_memory_usage > 2048]  # 2GB
        if high_memory_scenarios:
            recommendations.append(f"Review memory usage in {len(high_memory_scenarios)} scenarios (>2GB)")
        
        # Failure recommendations
        failed_scenarios = [r for r in results if not r.success]
        if failed_scenarios:
            recommendations.append(f"Investigate {len(failed_scenarios)} failed scenarios")
        
        return recommendations
    
    def _save_validation_report(self, report: E2EValidationReport):
        """Save validation report to file."""
        report_file = self.workspace_dir / f"{report.report_id}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
        
        logger.info(f"Validation report saved: {report_file}")
