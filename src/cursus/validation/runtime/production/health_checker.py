"""
Health Check System for Pipeline Runtime Testing.

This module provides comprehensive health checks for production deployment,
system validation, and operational monitoring.
"""

import os
import time
import psutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field, field_validator
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentHealthCheck(BaseModel):
    """Health check result for a single component."""
    
    component_name: str = Field(..., description="Name of the component")
    status: HealthStatus = Field(..., description="Health status")
    message: str = Field(..., description="Status message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    check_duration: float = Field(..., description="Check duration in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for issues")


class SystemHealthReport(BaseModel):
    """Comprehensive system health report."""
    
    report_id: str
    generation_time: datetime
    overall_status: HealthStatus
    total_checks: int
    healthy_checks: int
    warning_checks: int
    critical_checks: int
    component_results: List[ComponentHealthCheck]
    system_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if self.total_checks == 0:
            return 0.0
        
        # Weight the scores: healthy=100, warning=50, critical=0
        score = (self.healthy_checks * 100 + self.warning_checks * 50) / self.total_checks
        return round(score, 2)


class HealthChecker:
    """
    Comprehensive health check system for pipeline runtime testing.
    
    Provides system validation, dependency checks, and operational monitoring
    for production deployment readiness.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the health checker.
        
        Args:
            config: Configuration dictionary for health checks
        """
        self.config = config or {}
        self.check_timeout = self.config.get('check_timeout', 30.0)  # seconds
        self.workspace_dir = Path(self.config.get('workspace_dir', './health_check_workspace'))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Health Checker initialized")
    
    def check_system_health(self) -> SystemHealthReport:
        """
        Perform comprehensive system health check.
        
        Returns:
            Complete system health report
        """
        logger.info("Starting comprehensive system health check")
        
        start_time = time.time()
        component_results = []
        
        # Core component checks
        component_results.append(self._check_core_components())
        component_results.append(self._check_dependencies())
        component_results.append(self._check_workspace_access())
        component_results.append(self._check_aws_access())
        component_results.append(self._check_performance())
        component_results.append(self._check_disk_space())
        component_results.append(self._check_memory_availability())
        component_results.append(self._check_python_environment())
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(component_results)
        
        # Count status types
        status_counts = self._count_status_types(component_results)
        
        # Generate system metrics
        system_metrics = self._collect_system_metrics()
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(component_results)
        
        report = SystemHealthReport(
            report_id=f"health_check_{int(time.time())}",
            generation_time=datetime.now(),
            overall_status=overall_status,
            total_checks=len(component_results),
            healthy_checks=status_counts['healthy'],
            warning_checks=status_counts['warning'],
            critical_checks=status_counts['critical'],
            component_results=component_results,
            system_metrics=system_metrics,
            recommendations=recommendations
        )
        
        total_duration = time.time() - start_time
        logger.info(f"System health check completed in {total_duration:.2f}s - Status: {overall_status}")
        
        return report
    
    def _check_core_components(self) -> ComponentHealthCheck:
        """Check core pipeline runtime components."""
        start_time = time.time()
        
        try:
            # Check if core modules can be imported
            from ..core.pipeline_script_executor import PipelineScriptExecutor
            from ..execution.pipeline_executor import PipelineExecutor
            from ..integration.real_data_tester import RealDataTester
            
            # Try to instantiate core components
            executor = PipelineScriptExecutor()
            pipeline_executor = PipelineExecutor()
            data_tester = RealDataTester()
            
            return ComponentHealthCheck(
                component_name="core_components",
                status=HealthStatus.HEALTHY,
                message="All core components are accessible and functional",
                details={
                    "pipeline_script_executor": "OK",
                    "pipeline_executor": "OK",
                    "real_data_tester": "OK"
                },
                check_duration=time.time() - start_time
            )
            
        except ImportError as e:
            return ComponentHealthCheck(
                component_name="core_components",
                status=HealthStatus.CRITICAL,
                message=f"Core component import failed: {e}",
                details={"error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Check Python path and module installation"]
            )
        except Exception as e:
            return ComponentHealthCheck(
                component_name="core_components",
                status=HealthStatus.WARNING,
                message=f"Core component instantiation issue: {e}",
                details={"error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Review component configuration"]
            )
    
    def _check_dependencies(self) -> ComponentHealthCheck:
        """Check system dependencies and required packages."""
        start_time = time.time()
        
        required_packages = [
            'pydantic', 'psutil', 'boto3', 'yaml', 'pandas', 'numpy'
        ]
        
        missing_packages = []
        package_versions = {}
        
        for package in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                package_versions[package] = version
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return ComponentHealthCheck(
                component_name="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Missing required packages: {', '.join(missing_packages)}",
                details={
                    "missing_packages": missing_packages,
                    "available_packages": package_versions
                },
                check_duration=time.time() - start_time,
                recommendations=[f"Install missing packages: pip install {' '.join(missing_packages)}"]
            )
        else:
            return ComponentHealthCheck(
                component_name="dependencies",
                status=HealthStatus.HEALTHY,
                message="All required dependencies are available",
                details={"package_versions": package_versions},
                check_duration=time.time() - start_time
            )
    
    def _check_workspace_access(self) -> ComponentHealthCheck:
        """Check workspace directory access and permissions."""
        start_time = time.time()
        
        try:
            # Test directory creation
            test_dir = self.workspace_dir / "health_check_test"
            test_dir.mkdir(exist_ok=True)
            
            # Test file write
            test_file = test_dir / "test_file.txt"
            test_file.write_text("health check test")
            
            # Test file read
            content = test_file.read_text()
            
            # Cleanup
            test_file.unlink()
            test_dir.rmdir()
            
            return ComponentHealthCheck(
                component_name="workspace_access",
                status=HealthStatus.HEALTHY,
                message="Workspace directory is accessible with full permissions",
                details={
                    "workspace_path": str(self.workspace_dir),
                    "permissions": "read/write/execute"
                },
                check_duration=time.time() - start_time
            )
            
        except PermissionError as e:
            return ComponentHealthCheck(
                component_name="workspace_access",
                status=HealthStatus.CRITICAL,
                message=f"Workspace permission error: {e}",
                details={"workspace_path": str(self.workspace_dir), "error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Check directory permissions", "Ensure write access to workspace"]
            )
        except Exception as e:
            return ComponentHealthCheck(
                component_name="workspace_access",
                status=HealthStatus.WARNING,
                message=f"Workspace access issue: {e}",
                details={"workspace_path": str(self.workspace_dir), "error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Verify workspace directory configuration"]
            )
    
    def _check_aws_access(self) -> ComponentHealthCheck:
        """Check AWS credentials and service access."""
        start_time = time.time()
        
        try:
            # Check AWS credentials
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if not credentials:
                return ComponentHealthCheck(
                    component_name="aws_access",
                    status=HealthStatus.WARNING,
                    message="AWS credentials not configured",
                    details={"credentials": "not_found"},
                    check_duration=time.time() - start_time,
                    recommendations=["Configure AWS credentials", "Set up AWS CLI or environment variables"]
                )
            
            # Test S3 access
            s3_client = boto3.client('s3')
            try:
                s3_client.list_buckets()
                s3_status = "accessible"
            except ClientError as e:
                s3_status = f"error: {e}"
            
            # Test SageMaker access
            sagemaker_client = boto3.client('sagemaker')
            try:
                sagemaker_client.list_training_jobs(MaxResults=1)
                sagemaker_status = "accessible"
            except ClientError as e:
                sagemaker_status = f"error: {e}"
            
            # Determine overall AWS status
            if "error" in s3_status or "error" in sagemaker_status:
                status = HealthStatus.WARNING
                message = "AWS credentials configured but some services have access issues"
            else:
                status = HealthStatus.HEALTHY
                message = "AWS credentials configured and services accessible"
            
            return ComponentHealthCheck(
                component_name="aws_access",
                status=status,
                message=message,
                details={
                    "credentials": "configured",
                    "s3_access": s3_status,
                    "sagemaker_access": sagemaker_status,
                    "region": session.region_name or "default"
                },
                check_duration=time.time() - start_time,
                recommendations=["Review AWS IAM permissions"] if status == HealthStatus.WARNING else []
            )
            
        except NoCredentialsError:
            return ComponentHealthCheck(
                component_name="aws_access",
                status=HealthStatus.WARNING,
                message="AWS credentials not found",
                details={"credentials": "not_found"},
                check_duration=time.time() - start_time,
                recommendations=["Configure AWS credentials", "Set up AWS CLI or environment variables"]
            )
        except Exception as e:
            return ComponentHealthCheck(
                component_name="aws_access",
                status=HealthStatus.WARNING,
                message=f"AWS access check failed: {e}",
                details={"error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Check AWS configuration", "Verify network connectivity"]
            )
    
    def _check_performance(self) -> ComponentHealthCheck:
        """Check system performance metrics."""
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status based on thresholds
            issues = []
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > 90:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            if disk_percent > 90:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            if issues:
                status = HealthStatus.CRITICAL if len(issues) > 1 else HealthStatus.WARNING
                message = f"Performance issues detected: {'; '.join(issues)}"
                recommendations = ["Monitor resource usage", "Consider scaling resources"]
            else:
                status = HealthStatus.HEALTHY
                message = "System performance is within normal ranges"
                recommendations = []
            
            return ComponentHealthCheck(
                component_name="performance",
                status=status,
                message=message,
                details={
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory_percent,
                    "disk_usage_percent": disk_percent,
                    "available_memory_gb": memory.available / (1024**3)
                },
                check_duration=time.time() - start_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ComponentHealthCheck(
                component_name="performance",
                status=HealthStatus.WARNING,
                message=f"Performance check failed: {e}",
                details={"error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Check system monitoring tools"]
            )
    
    def _check_disk_space(self) -> ComponentHealthCheck:
        """Check available disk space."""
        start_time = time.time()
        
        try:
            disk_usage = psutil.disk_usage(str(self.workspace_dir))
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            if free_gb < 1.0:  # Less than 1GB free
                status = HealthStatus.CRITICAL
                message = f"Critical: Only {free_gb:.2f}GB disk space remaining"
                recommendations = ["Free up disk space immediately", "Consider expanding storage"]
            elif free_gb < 5.0:  # Less than 5GB free
                status = HealthStatus.WARNING
                message = f"Warning: Only {free_gb:.2f}GB disk space remaining"
                recommendations = ["Monitor disk usage", "Plan for storage expansion"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Sufficient disk space available: {free_gb:.2f}GB free"
                recommendations = []
            
            return ComponentHealthCheck(
                component_name="disk_space",
                status=status,
                message=message,
                details={
                    "free_space_gb": free_gb,
                    "total_space_gb": total_gb,
                    "used_percent": used_percent,
                    "workspace_path": str(self.workspace_dir)
                },
                check_duration=time.time() - start_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ComponentHealthCheck(
                component_name="disk_space",
                status=HealthStatus.WARNING,
                message=f"Disk space check failed: {e}",
                details={"error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Check disk access permissions"]
            )
    
    def _check_memory_availability(self) -> ComponentHealthCheck:
        """Check memory availability for pipeline execution."""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            # Minimum memory requirements for pipeline execution
            min_required_gb = 2.0
            recommended_gb = 4.0
            
            if available_gb < min_required_gb:
                status = HealthStatus.CRITICAL
                message = f"Insufficient memory: {available_gb:.2f}GB available (minimum {min_required_gb}GB required)"
                recommendations = ["Free up memory", "Close unnecessary applications", "Consider adding more RAM"]
            elif available_gb < recommended_gb:
                status = HealthStatus.WARNING
                message = f"Limited memory: {available_gb:.2f}GB available (recommended {recommended_gb}GB)"
                recommendations = ["Monitor memory usage during pipeline execution", "Consider optimizing memory usage"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Sufficient memory available: {available_gb:.2f}GB"
                recommendations = []
            
            return ComponentHealthCheck(
                component_name="memory_availability",
                status=status,
                message=message,
                details={
                    "available_memory_gb": available_gb,
                    "total_memory_gb": total_gb,
                    "memory_usage_percent": memory.percent,
                    "minimum_required_gb": min_required_gb,
                    "recommended_gb": recommended_gb
                },
                check_duration=time.time() - start_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ComponentHealthCheck(
                component_name="memory_availability",
                status=HealthStatus.WARNING,
                message=f"Memory check failed: {e}",
                details={"error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Check system memory monitoring"]
            )
    
    def _check_python_environment(self) -> ComponentHealthCheck:
        """Check Python environment and version compatibility."""
        start_time = time.time()
        
        try:
            import sys
            python_version = sys.version
            python_version_info = sys.version_info
            
            # Check Python version compatibility
            min_version = (3, 8)
            recommended_version = (3, 9)
            
            if python_version_info[:2] < min_version:
                status = HealthStatus.CRITICAL
                message = f"Python version too old: {python_version_info[:2]} (minimum {min_version} required)"
                recommendations = ["Upgrade Python to a supported version"]
            elif python_version_info[:2] < recommended_version:
                status = HealthStatus.WARNING
                message = f"Python version acceptable but not optimal: {python_version_info[:2]} (recommended {recommended_version})"
                recommendations = ["Consider upgrading Python for better performance"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Python version is compatible: {python_version_info[:2]}"
                recommendations = []
            
            return ComponentHealthCheck(
                component_name="python_environment",
                status=status,
                message=message,
                details={
                    "python_version": python_version,
                    "version_info": python_version_info[:3],
                    "executable": sys.executable,
                    "platform": sys.platform
                },
                check_duration=time.time() - start_time,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ComponentHealthCheck(
                component_name="python_environment",
                status=HealthStatus.WARNING,
                message=f"Python environment check failed: {e}",
                details={"error": str(e)},
                check_duration=time.time() - start_time,
                recommendations=["Check Python installation"]
            )
    
    def _calculate_overall_status(self, component_results: List[ComponentHealthCheck]) -> HealthStatus:
        """Calculate overall system health status."""
        if not component_results:
            return HealthStatus.UNKNOWN
        
        # If any component is critical, overall status is critical
        if any(result.status == HealthStatus.CRITICAL for result in component_results):
            return HealthStatus.CRITICAL
        
        # If any component has warnings, overall status is warning
        if any(result.status == HealthStatus.WARNING for result in component_results):
            return HealthStatus.WARNING
        
        # If all components are healthy, overall status is healthy
        if all(result.status == HealthStatus.HEALTHY for result in component_results):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def _count_status_types(self, component_results: List[ComponentHealthCheck]) -> Dict[str, int]:
        """Count the number of each status type."""
        counts = {
            'healthy': 0,
            'warning': 0,
            'critical': 0,
            'unknown': 0
        }
        
        for result in component_results:
            counts[result.status.value] += 1
        
        return counts
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect additional system metrics."""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
                "network_connections": len(psutil.net_connections()),
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {"error": str(e)}
    
    def _generate_health_recommendations(self, component_results: List[ComponentHealthCheck]) -> List[str]:
        """Generate overall health recommendations."""
        recommendations = []
        
        # Collect all component recommendations
        for result in component_results:
            recommendations.extend(result.recommendations)
        
        # Add general recommendations based on overall status
        critical_count = sum(1 for r in component_results if r.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for r in component_results if r.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical issues before production deployment")
        
        if warning_count > 2:
            recommendations.append(f"Review and resolve {warning_count} warning conditions for optimal performance")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def save_health_report(self, report: SystemHealthReport, output_path: str = None) -> str:
        """
        Save health report to file.
        
        Args:
            report: System health report
            output_path: Output file path (optional)
            
        Returns:
            Path to saved report file
        """
        if output_path is None:
            output_path = f"health_report_{report.report_id}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            import json
            json.dump(report.dict(), f, indent=2, default=str)
        
        logger.info(f"Health report saved: {output_file}")
        return str(output_file)
