"""
Production CLI commands for Pipeline Runtime Testing System.

This module provides CLI commands for production readiness validation,
health checks, performance monitoring, and deployment validation.
"""

import click
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..validation.runtime.production.e2e_validator import EndToEndValidator, E2ETestScenario
from ..validation.runtime.production.performance_optimizer import PerformanceOptimizer, MonitoringConfig
from ..validation.runtime.production.health_checker import HealthChecker
from ..validation.runtime.production.deployment_validator import DeploymentValidator, DeploymentConfig, DeploymentEnvironment

logger = logging.getLogger(__name__)


@click.group(name='production')
def production_cli():
    """Production readiness commands for pipeline runtime testing."""
    pass


@production_cli.command('health-check')
@click.option('--config', '-c', type=click.Path(exists=True), help='Health check configuration file')
@click.option('--output', '-o', type=click.Path(), help='Output file for health report')
@click.option('--workspace', '-w', type=click.Path(), help='Workspace directory for health checks')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def health_check(config: Optional[str], output: Optional[str], workspace: Optional[str], verbose: bool):
    """Perform comprehensive system health check."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    click.echo("🏥 Starting system health check...")
    
    # Load configuration
    health_config = {}
    if config:
        with open(config, 'r') as f:
            health_config = json.load(f)
    
    if workspace:
        health_config['workspace_dir'] = workspace
    
    # Initialize health checker
    health_checker = HealthChecker(health_config)
    
    # Perform health check
    try:
        report = health_checker.check_system_health()
        
        # Display results
        click.echo(f"\n📊 Health Check Results:")
        click.echo(f"Overall Status: {_format_status(report.overall_status.value)}")
        click.echo(f"Health Score: {report.health_score}/100")
        click.echo(f"Total Checks: {report.total_checks}")
        click.echo(f"✅ Healthy: {report.healthy_checks}")
        click.echo(f"⚠️  Warnings: {report.warning_checks}")
        click.echo(f"❌ Critical: {report.critical_checks}")
        
        # Show component details
        click.echo(f"\n🔍 Component Details:")
        for result in report.component_results:
            status_icon = _get_status_icon(result.status.value)
            click.echo(f"{status_icon} {result.component_name}: {result.message}")
        
        # Show recommendations
        if report.recommendations:
            click.echo(f"\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                click.echo(f"{i}. {rec}")
        
        # Save report
        if output:
            report_path = health_checker.save_health_report(report, output)
            click.echo(f"\n📄 Health report saved: {report_path}")
        
        # Exit with appropriate code
        if report.overall_status.value == 'critical':
            click.echo("\n❌ System has critical issues - not ready for production")
            exit(1)
        elif report.overall_status.value == 'warning':
            click.echo("\n⚠️  System has warnings - review before production deployment")
            exit(2)
        else:
            click.echo("\n✅ System health check passed")
            
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}", err=True)
        exit(1)


@production_cli.command('validate-e2e')
@click.argument('scenarios_dir', type=click.Path(exists=True))
@click.option('--workspace', '-w', type=click.Path(), help='Workspace directory for E2E tests')
@click.option('--output', '-o', type=click.Path(), help='Output file for validation report')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def validate_e2e(scenarios_dir: str, workspace: Optional[str], output: Optional[str], verbose: bool):
    """Run end-to-end validation with test scenarios."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    click.echo("🧪 Starting end-to-end validation...")
    
    # Initialize E2E validator
    validator_config = {}
    if workspace:
        validator_config['workspace_dir'] = workspace
    
    validator = EndToEndValidator(workspace, validator_config)
    
    try:
        # Run comprehensive validation
        report = validator.run_comprehensive_validation(scenarios_dir)
        
        # Display results
        click.echo(f"\n📊 E2E Validation Results:")
        click.echo(f"Success Rate: {report.success_rate:.1%}")
        click.echo(f"Total Scenarios: {report.total_scenarios}")
        click.echo(f"✅ Successful: {report.successful_scenarios}")
        click.echo(f"❌ Failed: {report.failed_scenarios}")
        click.echo(f"⏱️  Total Time: {report.total_execution_time:.2f}s")
        click.echo(f"📈 Peak Memory: {report.peak_memory_usage:.1f}MB")
        
        # Show scenario details
        click.echo(f"\n🔍 Scenario Results:")
        for result in report.scenario_results:
            status_icon = "✅" if result.success else "❌"
            click.echo(f"{status_icon} {result.scenario_name}: {result.total_duration:.2f}s, {result.peak_memory_usage:.1f}MB")
            if result.warnings:
                for warning in result.warnings:
                    click.echo(f"    ⚠️  {warning}")
        
        # Show recommendations
        if report.recommendations:
            click.echo(f"\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                click.echo(f"{i}. {rec}")
        
        # Save report if requested
        if output:
            with open(output, 'w') as f:
                json.dump(report.dict(), f, indent=2, default=str)
            click.echo(f"\n📄 E2E validation report saved: {output}")
        
        # Exit with appropriate code
        if report.success_rate < 0.8:
            click.echo("\n❌ E2E validation failed - success rate below 80%")
            exit(1)
        elif report.success_rate < 1.0:
            click.echo("\n⚠️  E2E validation passed with warnings")
            exit(2)
        else:
            click.echo("\n✅ E2E validation passed successfully")
            
    except Exception as e:
        click.echo(f"❌ E2E validation failed: {e}", err=True)
        exit(1)


@production_cli.command('monitor-performance')
@click.option('--duration', '-d', type=int, default=60, help='Monitoring duration in seconds')
@click.option('--interval', '-i', type=float, default=1.0, help='Monitoring interval in seconds')
@click.option('--output', '-o', type=click.Path(), help='Output file for performance report')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def monitor_performance(duration: int, interval: float, output: Optional[str], verbose: bool):
    """Monitor system performance and generate optimization recommendations."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    click.echo(f"📊 Starting performance monitoring for {duration}s...")
    
    # Initialize performance optimizer
    config = MonitoringConfig(interval_seconds=interval)
    optimizer = PerformanceOptimizer(config)
    
    try:
        # Start monitoring
        optimizer.start_monitoring()
        click.echo(f"⏱️  Monitoring system performance...")
        
        # Show progress
        import time
        for i in range(duration):
            if i % 10 == 0:
                click.echo(f"📈 Monitoring... {i}/{duration}s", nl=False)
            time.sleep(1)
            if i % 10 == 9:
                click.echo()
        
        # Stop monitoring and analyze
        click.echo(f"\n🔍 Analyzing performance data...")
        summary = optimizer.stop_monitoring()
        report = optimizer.analyze_performance()
        
        # Display results
        click.echo(f"\n📊 Performance Analysis Results:")
        click.echo(f"Monitoring Duration: {report.analysis_duration:.2f}s")
        click.echo(f"Total Samples: {report.total_samples}")
        click.echo(f"Average CPU: {report.average_metrics.cpu_usage_percent:.1f}%")
        click.echo(f"Average Memory: {report.average_metrics.memory_usage_mb:.1f}MB")
        click.echo(f"Peak Memory: {report.peak_metrics.memory_usage_mb:.1f}MB")
        
        # Show performance trends
        if 'insufficient_data' not in report.performance_trends:
            trends = report.performance_trends
            click.echo(f"\n📈 Performance Trends:")
            click.echo(f"CPU Trend: {trends['cpu_usage_trend']['direction']}")
            click.echo(f"Memory Trend: {trends['memory_usage_trend']['direction']}")
            click.echo(f"Stability Score: {trends['performance_stability']:.1f}/100")
        
        # Show optimization recommendations
        if report.recommendations:
            click.echo(f"\n💡 Optimization Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                severity_icon = _get_severity_icon(rec.severity)
                click.echo(f"{i}. {severity_icon} [{rec.category.upper()}] {rec.description}")
                click.echo(f"   Action: {rec.suggested_action}")
                click.echo(f"   Impact: {rec.estimated_improvement}")
        
        # Save report
        if output:
            report_path = optimizer.save_performance_report(report, output)
            click.echo(f"\n📄 Performance report saved: {report_path}")
        
        # Show optimized parameters
        optimized_params = optimizer.optimize_execution_parameters()
        if optimized_params:
            click.echo(f"\n⚙️  Optimized Parameters:")
            for param, value in optimized_params.items():
                click.echo(f"  {param}: {value}")
        
        click.echo("\n✅ Performance monitoring completed")
        
    except Exception as e:
        click.echo(f"❌ Performance monitoring failed: {e}", err=True)
        exit(1)


@production_cli.command('validate-deployment')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for deployment report')
@click.option('--generate-manifests', is_flag=True, help='Generate Kubernetes manifests')
@click.option('--generate-compose', is_flag=True, help='Generate Docker Compose file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def validate_deployment(config_file: str, output: Optional[str], generate_manifests: bool, 
                       generate_compose: bool, verbose: bool):
    """Validate deployment configuration and readiness."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    click.echo("🚀 Starting deployment validation...")
    
    try:
        # Load deployment configuration
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        deployment_config = DeploymentConfig(**config_data)
        
        # Initialize deployment validator
        validator = DeploymentValidator()
        
        # Validate deployment
        report = validator.validate_deployment(deployment_config)
        
        # Display results
        click.echo(f"\n📊 Deployment Validation Results:")
        click.echo(f"Environment: {report.environment.value}")
        click.echo(f"Overall Status: {_format_status(report.overall_status.value)}")
        click.echo(f"Readiness Score: {report.readiness_score}/100")
        
        # Show validation results
        click.echo(f"\n🔍 Component Validation:")
        for result in report.validation_results:
            status_icon = _get_status_icon(result.status.value)
            click.echo(f"{status_icon} {result.component}: {result.message}")
        
        # Show blockers
        if report.blockers:
            click.echo(f"\n🚫 Deployment Blockers:")
            for i, blocker in enumerate(report.blockers, 1):
                click.echo(f"{i}. {blocker}")
        
        # Show warnings
        if report.warnings:
            click.echo(f"\n⚠️  Deployment Warnings:")
            for i, warning in enumerate(report.warnings, 1):
                click.echo(f"{i}. {warning}")
        
        # Show recommendations
        if report.recommendations:
            click.echo(f"\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                click.echo(f"{i}. {rec}")
        
        # Generate manifests if requested
        if generate_manifests:
            click.echo(f"\n📝 Generating Kubernetes manifests...")
            manifests = validator.generate_kubernetes_manifests(deployment_config)
            
            manifest_dir = Path("./k8s-manifests")
            manifest_dir.mkdir(exist_ok=True)
            
            for filename, content in manifests.items():
                manifest_path = manifest_dir / filename
                manifest_path.write_text(content)
                click.echo(f"  Generated: {manifest_path}")
        
        # Generate Docker Compose if requested
        if generate_compose:
            click.echo(f"\n🐳 Generating Docker Compose file...")
            compose_content = validator.generate_docker_compose(deployment_config)
            
            compose_path = Path("./docker-compose.yml")
            compose_path.write_text(compose_content)
            click.echo(f"  Generated: {compose_path}")
        
        # Save report
        if output:
            report_path = validator.save_deployment_report(report, output)
            click.echo(f"\n📄 Deployment report saved: {report_path}")
        
        # Exit with appropriate code
        if report.overall_status.value == 'not_ready':
            click.echo("\n❌ Deployment validation failed - not ready for deployment")
            exit(1)
        elif report.overall_status.value == 'needs_review':
            click.echo("\n⚠️  Deployment validation passed with warnings - review before deployment")
            exit(2)
        else:
            click.echo("\n✅ Deployment validation passed - ready for deployment")
            
    except Exception as e:
        click.echo(f"❌ Deployment validation failed: {e}", err=True)
        exit(1)


@production_cli.command('validate-system')
@click.argument('scenarios_dir', type=click.Path(exists=True))
@click.option('--deployment-config', type=click.Path(exists=True), help='Deployment configuration file')
@click.option('--workspace', '-w', type=click.Path(), help='Workspace directory')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory for all reports')
@click.option('--performance-duration', type=int, default=30, help='Performance monitoring duration')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def validate_system(scenarios_dir: str, deployment_config: Optional[str], workspace: Optional[str], 
                   output_dir: Optional[str], performance_duration: int, verbose: bool):
    """Run comprehensive system validation (health, E2E, performance, deployment)."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    click.echo("🔍 Starting comprehensive system validation...")
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path("./validation_reports")
        output_path.mkdir(exist_ok=True)
    
    validation_results = {}
    overall_success = True
    
    try:
        # 1. Health Check
        click.echo("\n🏥 Step 1/4: System Health Check")
        health_checker = HealthChecker({'workspace_dir': workspace} if workspace else {})
        health_report = health_checker.check_system_health()
        
        health_file = output_path / "health_report.json"
        health_checker.save_health_report(health_report, str(health_file))
        
        validation_results['health'] = {
            'status': health_report.overall_status.value,
            'score': health_report.health_score,
            'report_file': str(health_file)
        }
        
        if health_report.overall_status.value == 'critical':
            overall_success = False
        
        click.echo(f"  Status: {_format_status(health_report.overall_status.value)} (Score: {health_report.health_score}/100)")
        
        # 2. E2E Validation
        click.echo("\n🧪 Step 2/4: End-to-End Validation")
        e2e_validator = EndToEndValidator(workspace)
        e2e_report = e2e_validator.run_comprehensive_validation(scenarios_dir)
        
        e2e_file = output_path / "e2e_report.json"
        with open(e2e_file, 'w') as f:
            json.dump(e2e_report.dict(), f, indent=2, default=str)
        
        validation_results['e2e'] = {
            'success_rate': e2e_report.success_rate,
            'scenarios': e2e_report.total_scenarios,
            'report_file': str(e2e_file)
        }
        
        if e2e_report.success_rate < 0.8:
            overall_success = False
        
        click.echo(f"  Success Rate: {e2e_report.success_rate:.1%} ({e2e_report.successful_scenarios}/{e2e_report.total_scenarios})")
        
        # 3. Performance Monitoring
        click.echo(f"\n📊 Step 3/4: Performance Monitoring ({performance_duration}s)")
        optimizer = PerformanceOptimizer()
        optimizer.start_monitoring()
        
        import time
        time.sleep(performance_duration)
        
        optimizer.stop_monitoring()
        perf_report = optimizer.analyze_performance()
        
        perf_file = output_path / "performance_report.json"
        optimizer.save_performance_report(perf_report, str(perf_file))
        
        validation_results['performance'] = {
            'avg_cpu': perf_report.average_metrics.cpu_usage_percent,
            'peak_memory': perf_report.peak_metrics.memory_usage_mb,
            'recommendations': len(perf_report.recommendations),
            'report_file': str(perf_file)
        }
        
        click.echo(f"  CPU: {perf_report.average_metrics.cpu_usage_percent:.1f}%, Memory: {perf_report.peak_metrics.memory_usage_mb:.1f}MB")
        
        # 4. Deployment Validation (if config provided)
        if deployment_config:
            click.echo("\n🚀 Step 4/4: Deployment Validation")
            
            with open(deployment_config, 'r') as f:
                config_data = json.load(f)
            
            deploy_config = DeploymentConfig(**config_data)
            deploy_validator = DeploymentValidator()
            deploy_report = deploy_validator.validate_deployment(deploy_config)
            
            deploy_file = output_path / "deployment_report.json"
            deploy_validator.save_deployment_report(deploy_report, str(deploy_file))
            
            validation_results['deployment'] = {
                'status': deploy_report.overall_status.value,
                'readiness_score': deploy_report.readiness_score,
                'blockers': len(deploy_report.blockers),
                'report_file': str(deploy_file)
            }
            
            if deploy_report.overall_status.value == 'not_ready':
                overall_success = False
            
            click.echo(f"  Status: {_format_status(deploy_report.overall_status.value)} (Score: {deploy_report.readiness_score}/100)")
        else:
            click.echo("\n⏭️  Step 4/4: Deployment Validation (Skipped - no config provided)")
        
        # Generate summary report
        summary_file = output_path / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': str(datetime.now()),
                'overall_success': overall_success,
                'validation_results': validation_results
            }, f, indent=2)
        
        # Display final results
        click.echo(f"\n📋 Comprehensive Validation Summary:")
        click.echo(f"Overall Status: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        click.echo(f"Reports saved to: {output_path}")
        
        for component, results in validation_results.items():
            click.echo(f"  {component.upper()}: {results}")
        
        if overall_success:
            click.echo("\n🎉 System validation completed successfully - ready for production!")
        else:
            click.echo("\n⚠️  System validation completed with issues - review reports before production")
            exit(1)
            
    except Exception as e:
        click.echo(f"❌ System validation failed: {e}", err=True)
        exit(1)


def _format_status(status: str) -> str:
    """Format status with appropriate styling."""
    status_map = {
        'healthy': '✅ HEALTHY',
        'warning': '⚠️  WARNING',
        'critical': '❌ CRITICAL',
        'ready': '✅ READY',
        'not_ready': '❌ NOT READY',
        'needs_review': '⚠️  NEEDS REVIEW',
        'unknown': '❓ UNKNOWN'
    }
    return status_map.get(status, status.upper())


def _get_status_icon(status: str) -> str:
    """Get icon for status."""
    icon_map = {
        'healthy': '✅',
        'warning': '⚠️',
        'critical': '❌',
        'ready': '✅',
        'not_ready': '❌',
        'needs_review': '⚠️',
        'unknown': '❓'
    }
    return icon_map.get(status, '❓')


def _get_severity_icon(severity: str) -> str:
    """Get icon for severity level."""
    severity_map = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🟠',
        'critical': '🔴'
    }
    return severity_map.get(severity, '⚪')


if __name__ == '__main__':
    production_cli()
