"""
Deployment Validator for Pipeline Runtime Testing.

This module provides deployment configuration validation,
CI/CD integration, and production deployment readiness checks.
"""

import os
import json
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DeploymentEnvironment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class DeploymentStatus(str, Enum):
    """Deployment validation status."""
    READY = "ready"
    NOT_READY = "not_ready"
    NEEDS_REVIEW = "needs_review"
    UNKNOWN = "unknown"


class DeploymentConfig(BaseModel):
    """Deployment configuration model."""
    
    environment: DeploymentEnvironment = Field(..., description="Target deployment environment")
    docker_image: str = Field(..., description="Docker image name and tag")
    resource_limits: Dict[str, str] = Field(..., description="Resource limits (memory, cpu)")
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    health_check_endpoint: str = Field(default="/health", description="Health check endpoint")
    port: int = Field(default=8080, description="Application port")
    replicas: int = Field(default=1, description="Number of replicas")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        if v < 1 or v > 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator('replicas')
    @classmethod
    def validate_replicas(cls, v):
        if v < 1 or v > 100:
            raise ValueError("Replicas must be between 1 and 100")
        return v


class DeploymentValidationResult(BaseModel):
    """Result of deployment validation."""
    
    component: str = Field(..., description="Component being validated")
    status: DeploymentStatus = Field(..., description="Validation status")
    message: str = Field(..., description="Validation message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Validation details")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    validation_time: datetime = Field(default_factory=datetime.now, description="Validation timestamp")


class DeploymentReport(BaseModel):
    """Comprehensive deployment validation report."""
    
    report_id: str
    generation_time: datetime
    environment: DeploymentEnvironment
    overall_status: DeploymentStatus
    validation_results: List[DeploymentValidationResult]
    deployment_config: DeploymentConfig
    readiness_score: float = Field(..., description="Deployment readiness score (0-100)")
    blockers: List[str] = Field(default_factory=list, description="Deployment blockers")
    warnings: List[str] = Field(default_factory=list, description="Deployment warnings")
    recommendations: List[str] = Field(default_factory=list, description="Overall recommendations")


class DeploymentValidator:
    """
    Deployment validation system for pipeline runtime testing.
    
    Validates deployment configurations, Docker images, Kubernetes manifests,
    and CI/CD pipeline integration for production readiness.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the deployment validator.
        
        Args:
            config: Configuration dictionary for deployment validation
        """
        self.config = config or {}
        self.workspace_dir = Path(self.config.get('workspace_dir', './deployment_validation'))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Deployment Validator initialized")
    
    def validate_deployment(self, deployment_config: DeploymentConfig) -> DeploymentReport:
        """
        Validate deployment configuration and readiness.
        
        Args:
            deployment_config: Deployment configuration to validate
            
        Returns:
            Comprehensive deployment validation report
        """
        logger.info(f"Starting deployment validation for {deployment_config.environment}")
        
        validation_results = []
        
        # Core validation checks
        validation_results.append(self._validate_docker_configuration(deployment_config))
        validation_results.append(self._validate_kubernetes_configuration(deployment_config))
        validation_results.append(self._validate_resource_limits(deployment_config))
        validation_results.append(self._validate_environment_variables(deployment_config))
        validation_results.append(self._validate_health_checks(deployment_config))
        validation_results.append(self._validate_security_configuration(deployment_config))
        validation_results.append(self._validate_monitoring_setup(deployment_config))
        validation_results.append(self._validate_backup_strategy(deployment_config))
        
        # Calculate overall status and metrics
        overall_status = self._calculate_overall_deployment_status(validation_results)
        readiness_score = self._calculate_readiness_score(validation_results)
        blockers = self._identify_deployment_blockers(validation_results)
        warnings = self._identify_deployment_warnings(validation_results)
        recommendations = self._generate_deployment_recommendations(validation_results)
        
        report = DeploymentReport(
            report_id=f"deployment_validation_{int(datetime.now().timestamp())}",
            generation_time=datetime.now(),
            environment=deployment_config.environment,
            overall_status=overall_status,
            validation_results=validation_results,
            deployment_config=deployment_config,
            readiness_score=readiness_score,
            blockers=blockers,
            warnings=warnings,
            recommendations=recommendations
        )
        
        logger.info(f"Deployment validation completed - Status: {overall_status}, Score: {readiness_score}")
        
        return report
    
    def _validate_docker_configuration(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate Docker configuration and image."""
        try:
            # Check if Docker is available
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return DeploymentValidationResult(
                    component="docker_configuration",
                    status=DeploymentStatus.NOT_READY,
                    message="Docker is not available or not installed",
                    recommendations=["Install Docker", "Ensure Docker daemon is running"]
                )
            
            # Validate image name format
            if not self._is_valid_docker_image_name(config.docker_image):
                return DeploymentValidationResult(
                    component="docker_configuration",
                    status=DeploymentStatus.NOT_READY,
                    message=f"Invalid Docker image name format: {config.docker_image}",
                    recommendations=["Use valid Docker image naming convention (registry/repository:tag)"]
                )
            
            # Check if image exists locally or can be pulled
            image_status = self._check_docker_image_availability(config.docker_image)
            
            return DeploymentValidationResult(
                component="docker_configuration",
                status=DeploymentStatus.READY if image_status['available'] else DeploymentStatus.NEEDS_REVIEW,
                message=image_status['message'],
                details={
                    "docker_version": result.stdout.strip(),
                    "image_name": config.docker_image,
                    "image_available": image_status['available']
                },
                recommendations=image_status.get('recommendations', [])
            )
            
        except subprocess.TimeoutExpired:
            return DeploymentValidationResult(
                component="docker_configuration",
                status=DeploymentStatus.NOT_READY,
                message="Docker command timed out",
                recommendations=["Check Docker installation and daemon status"]
            )
        except Exception as e:
            return DeploymentValidationResult(
                component="docker_configuration",
                status=DeploymentStatus.NOT_READY,
                message=f"Docker validation failed: {e}",
                recommendations=["Check Docker installation and configuration"]
            )
    
    def _validate_kubernetes_configuration(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate Kubernetes configuration and connectivity."""
        try:
            # Check if kubectl is available
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return DeploymentValidationResult(
                    component="kubernetes_configuration",
                    status=DeploymentStatus.NEEDS_REVIEW,
                    message="kubectl is not available",
                    recommendations=["Install kubectl", "Configure Kubernetes cluster access"]
                )
            
            # Check cluster connectivity
            cluster_result = subprocess.run(['kubectl', 'cluster-info'], 
                                          capture_output=True, text=True, timeout=15)
            
            cluster_accessible = cluster_result.returncode == 0
            
            # Validate namespace
            namespace_status = self._validate_kubernetes_namespace(config.namespace)
            
            status = DeploymentStatus.READY if cluster_accessible and namespace_status['valid'] else DeploymentStatus.NEEDS_REVIEW
            
            return DeploymentValidationResult(
                component="kubernetes_configuration",
                status=status,
                message=f"Kubernetes cluster {'accessible' if cluster_accessible else 'not accessible'}, namespace {namespace_status['message']}",
                details={
                    "kubectl_version": result.stdout.strip(),
                    "cluster_accessible": cluster_accessible,
                    "namespace": config.namespace,
                    "namespace_valid": namespace_status['valid']
                },
                recommendations=self._get_kubernetes_recommendations(cluster_accessible, namespace_status)
            )
            
        except subprocess.TimeoutExpired:
            return DeploymentValidationResult(
                component="kubernetes_configuration",
                status=DeploymentStatus.NOT_READY,
                message="Kubernetes commands timed out",
                recommendations=["Check Kubernetes cluster connectivity", "Verify network access"]
            )
        except Exception as e:
            return DeploymentValidationResult(
                component="kubernetes_configuration",
                status=DeploymentStatus.NEEDS_REVIEW,
                message=f"Kubernetes validation failed: {e}",
                recommendations=["Check kubectl installation and cluster configuration"]
            )
    
    def _validate_resource_limits(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate resource limits configuration."""
        try:
            resource_limits = config.resource_limits
            issues = []
            recommendations = []
            
            # Validate memory limits
            if 'memory' in resource_limits:
                memory_limit = resource_limits['memory']
                if not self._is_valid_memory_limit(memory_limit):
                    issues.append(f"Invalid memory limit format: {memory_limit}")
                    recommendations.append("Use valid memory format (e.g., '2Gi', '1024Mi')")
                elif self._parse_memory_limit(memory_limit) < 512 * 1024 * 1024:  # 512Mi
                    issues.append("Memory limit too low (minimum 512Mi recommended)")
                    recommendations.append("Increase memory limit to at least 512Mi")
            else:
                issues.append("Memory limit not specified")
                recommendations.append("Specify memory limit for production deployment")
            
            # Validate CPU limits
            if 'cpu' in resource_limits:
                cpu_limit = resource_limits['cpu']
                if not self._is_valid_cpu_limit(cpu_limit):
                    issues.append(f"Invalid CPU limit format: {cpu_limit}")
                    recommendations.append("Use valid CPU format (e.g., '1000m', '0.5')")
            else:
                issues.append("CPU limit not specified")
                recommendations.append("Specify CPU limit for production deployment")
            
            status = DeploymentStatus.NOT_READY if issues else DeploymentStatus.READY
            message = f"Resource limits validation: {len(issues)} issues found" if issues else "Resource limits are properly configured"
            
            return DeploymentValidationResult(
                component="resource_limits",
                status=status,
                message=message,
                details={
                    "resource_limits": resource_limits,
                    "issues": issues
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DeploymentValidationResult(
                component="resource_limits",
                status=DeploymentStatus.NOT_READY,
                message=f"Resource limits validation failed: {e}",
                recommendations=["Review resource limits configuration"]
            )
    
    def _validate_environment_variables(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate environment variables configuration."""
        try:
            env_vars = config.environment_variables
            issues = []
            recommendations = []
            
            # Check for required environment variables based on environment
            required_vars = self._get_required_environment_variables(config.environment)
            
            for var in required_vars:
                if var not in env_vars:
                    issues.append(f"Required environment variable missing: {var}")
                    recommendations.append(f"Set {var} environment variable")
            
            # Check for sensitive data in environment variables
            sensitive_patterns = ['password', 'secret', 'key', 'token']
            for var_name, var_value in env_vars.items():
                if any(pattern in var_name.lower() for pattern in sensitive_patterns):
                    if len(var_value) < 10:  # Likely not a proper secret
                        issues.append(f"Potentially weak secret in {var_name}")
                        recommendations.append(f"Use strong secrets for {var_name}")
            
            status = DeploymentStatus.NEEDS_REVIEW if issues else DeploymentStatus.READY
            message = f"Environment variables validation: {len(issues)} issues found" if issues else "Environment variables are properly configured"
            
            return DeploymentValidationResult(
                component="environment_variables",
                status=status,
                message=message,
                details={
                    "environment_variables_count": len(env_vars),
                    "required_variables": required_vars,
                    "issues": issues
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DeploymentValidationResult(
                component="environment_variables",
                status=DeploymentStatus.NOT_READY,
                message=f"Environment variables validation failed: {e}",
                recommendations=["Review environment variables configuration"]
            )
    
    def _validate_health_checks(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate health check configuration."""
        try:
            health_endpoint = config.health_check_endpoint
            port = config.port
            
            issues = []
            recommendations = []
            
            # Validate health check endpoint format
            if not health_endpoint.startswith('/'):
                issues.append("Health check endpoint should start with '/'")
                recommendations.append("Use proper URL path format for health check endpoint")
            
            # Validate port configuration
            if port < 1024 and config.environment == DeploymentEnvironment.PRODUCTION:
                issues.append("Using privileged port in production")
                recommendations.append("Consider using non-privileged port (>1024) for production")
            
            # Check if health check implementation exists (basic check)
            health_check_implemented = self._check_health_check_implementation()
            
            if not health_check_implemented:
                issues.append("Health check endpoint implementation not found")
                recommendations.append("Implement health check endpoint in application")
            
            status = DeploymentStatus.NEEDS_REVIEW if issues else DeploymentStatus.READY
            message = f"Health checks validation: {len(issues)} issues found" if issues else "Health checks are properly configured"
            
            return DeploymentValidationResult(
                component="health_checks",
                status=status,
                message=message,
                details={
                    "health_check_endpoint": health_endpoint,
                    "port": port,
                    "health_check_implemented": health_check_implemented,
                    "issues": issues
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DeploymentValidationResult(
                component="health_checks",
                status=DeploymentStatus.NOT_READY,
                message=f"Health checks validation failed: {e}",
                recommendations=["Review health check configuration"]
            )
    
    def _validate_security_configuration(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate security configuration."""
        try:
            issues = []
            recommendations = []
            
            # Check for security-related environment variables
            env_vars = config.environment_variables
            
            # Check for proper secret management
            if config.environment == DeploymentEnvironment.PRODUCTION:
                if not any('SECRET' in key.upper() for key in env_vars.keys()):
                    issues.append("No secrets configuration found for production")
                    recommendations.append("Configure proper secret management for production")
            
            # Check for non-root user configuration
            if config.port < 1024:
                issues.append("Application may be running as root (privileged port)")
                recommendations.append("Configure application to run as non-root user")
            
            # Check for security context in resource limits
            if 'securityContext' not in config.resource_limits:
                issues.append("Security context not configured")
                recommendations.append("Configure security context for container")
            
            status = DeploymentStatus.NEEDS_REVIEW if issues else DeploymentStatus.READY
            message = f"Security validation: {len(issues)} issues found" if issues else "Security configuration is adequate"
            
            return DeploymentValidationResult(
                component="security_configuration",
                status=status,
                message=message,
                details={
                    "environment": config.environment.value,
                    "issues": issues
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DeploymentValidationResult(
                component="security_configuration",
                status=DeploymentStatus.NOT_READY,
                message=f"Security validation failed: {e}",
                recommendations=["Review security configuration"]
            )
    
    def _validate_monitoring_setup(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate monitoring and observability setup."""
        try:
            issues = []
            recommendations = []
            
            # Check for monitoring-related environment variables
            env_vars = config.environment_variables
            monitoring_vars = ['METRICS_PORT', 'PROMETHEUS_ENDPOINT', 'LOGGING_LEVEL']
            
            missing_monitoring = [var for var in monitoring_vars if var not in env_vars]
            if missing_monitoring:
                issues.append(f"Missing monitoring configuration: {', '.join(missing_monitoring)}")
                recommendations.append("Configure monitoring and metrics collection")
            
            # Check for production monitoring requirements
            if config.environment == DeploymentEnvironment.PRODUCTION:
                if 'LOGGING_LEVEL' not in env_vars or env_vars.get('LOGGING_LEVEL') == 'DEBUG':
                    issues.append("Debug logging enabled in production")
                    recommendations.append("Set appropriate logging level for production")
            
            status = DeploymentStatus.NEEDS_REVIEW if issues else DeploymentStatus.READY
            message = f"Monitoring validation: {len(issues)} issues found" if issues else "Monitoring setup is configured"
            
            return DeploymentValidationResult(
                component="monitoring_setup",
                status=status,
                message=message,
                details={
                    "monitoring_variables": [var for var in monitoring_vars if var in env_vars],
                    "issues": issues
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DeploymentValidationResult(
                component="monitoring_setup",
                status=DeploymentStatus.NOT_READY,
                message=f"Monitoring validation failed: {e}",
                recommendations=["Review monitoring configuration"]
            )
    
    def _validate_backup_strategy(self, config: DeploymentConfig) -> DeploymentValidationResult:
        """Validate backup and disaster recovery strategy."""
        try:
            issues = []
            recommendations = []
            
            # Check for backup-related configuration
            env_vars = config.environment_variables
            backup_vars = ['BACKUP_ENABLED', 'BACKUP_SCHEDULE', 'BACKUP_RETENTION']
            
            if config.environment == DeploymentEnvironment.PRODUCTION:
                missing_backup = [var for var in backup_vars if var not in env_vars]
                if missing_backup:
                    issues.append("Backup strategy not configured for production")
                    recommendations.append("Configure backup and disaster recovery strategy")
            
            # Check for data persistence configuration
            if 'PERSISTENT_STORAGE' not in env_vars and config.environment == DeploymentEnvironment.PRODUCTION:
                issues.append("Persistent storage not configured")
                recommendations.append("Configure persistent storage for production data")
            
            status = DeploymentStatus.NEEDS_REVIEW if issues else DeploymentStatus.READY
            message = f"Backup strategy validation: {len(issues)} issues found" if issues else "Backup strategy is configured"
            
            return DeploymentValidationResult(
                component="backup_strategy",
                status=status,
                message=message,
                details={
                    "backup_configured": any(var in env_vars for var in backup_vars),
                    "issues": issues
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return DeploymentValidationResult(
                component="backup_strategy",
                status=DeploymentStatus.NOT_READY,
                message=f"Backup validation failed: {e}",
                recommendations=["Review backup and recovery configuration"]
            )
    
    def _is_valid_docker_image_name(self, image_name: str) -> bool:
        """Validate Docker image name format."""
        # Basic validation for Docker image name format
        if not image_name or ':' not in image_name:
            return False
        
        parts = image_name.split(':')
        if len(parts) != 2:
            return False
        
        repository, tag = parts
        return bool(repository and tag and '/' in repository)
    
    def _check_docker_image_availability(self, image_name: str) -> Dict[str, Any]:
        """Check if Docker image is available."""
        try:
            # Try to inspect the image locally
            result = subprocess.run(['docker', 'image', 'inspect', image_name], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    'available': True,
                    'message': f"Docker image {image_name} is available locally"
                }
            else:
                return {
                    'available': False,
                    'message': f"Docker image {image_name} not found locally",
                    'recommendations': [f"Pull or build Docker image: {image_name}"]
                }
                
        except Exception as e:
            return {
                'available': False,
                'message': f"Failed to check Docker image availability: {e}",
                'recommendations': ["Check Docker configuration and image name"]
            }
    
    def _validate_kubernetes_namespace(self, namespace: str) -> Dict[str, Any]:
        """Validate Kubernetes namespace."""
        try:
            result = subprocess.run(['kubectl', 'get', 'namespace', namespace], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {'valid': True, 'message': 'exists'}
            else:
                return {'valid': False, 'message': 'does not exist'}
                
        except Exception:
            return {'valid': False, 'message': 'validation failed'}
    
    def _get_kubernetes_recommendations(self, cluster_accessible: bool, namespace_status: Dict[str, Any]) -> List[str]:
        """Get Kubernetes-specific recommendations."""
        recommendations = []
        
        if not cluster_accessible:
            recommendations.extend([
                "Configure kubectl to access Kubernetes cluster",
                "Verify cluster connectivity and credentials"
            ])
        
        if not namespace_status['valid']:
            recommendations.append(f"Create Kubernetes namespace or verify namespace name")
        
        return recommendations
    
    def _is_valid_memory_limit(self, memory_limit: str) -> bool:
        """Validate memory limit format."""
        import re
        pattern = r'^\d+(\.\d+)?(Mi|Gi|Ki|M|G|K)$'
        return bool(re.match(pattern, memory_limit))
    
    def _parse_memory_limit(self, memory_limit: str) -> int:
        """Parse memory limit to bytes."""
        import re
        match = re.match(r'^(\d+(?:\.\d+)?)(Mi|Gi|Ki|M|G|K)$', memory_limit)
        if not match:
            return 0
        
        value, unit = match.groups()
        value = float(value)
        
        multipliers = {
            'Ki': 1024, 'Mi': 1024**2, 'Gi': 1024**3,
            'K': 1000, 'M': 1000**2, 'G': 1000**3
        }
        
        return int(value * multipliers.get(unit, 1))
    
    def _is_valid_cpu_limit(self, cpu_limit: str) -> bool:
        """Validate CPU limit format."""
        import re
        # CPU can be specified as decimal (0.5) or millicores (500m)
        pattern = r'^(\d+(\.\d+)?|\d+m)$'
        return bool(re.match(pattern, cpu_limit))
    
    def _get_required_environment_variables(self, environment: DeploymentEnvironment) -> List[str]:
        """Get required environment variables for deployment environment."""
        base_vars = ['PYTHONPATH', 'LOG_LEVEL']
        
        if environment == DeploymentEnvironment.PRODUCTION:
            return base_vars + ['AWS_REGION', 'ENVIRONMENT']
        elif environment == DeploymentEnvironment.STAGING:
            return base_vars + ['ENVIRONMENT']
        else:
            return base_vars
    
    def _check_health_check_implementation(self) -> bool:
        """Check if health check endpoint is implemented."""
        # This is a placeholder - in real implementation, you would check
        # if the health check endpoint is implemented in the application code
        return True  # Assume implemented for now
    
    def _calculate_overall_deployment_status(self, results: List[DeploymentValidationResult]) -> DeploymentStatus:
        """Calculate overall deployment status."""
        if not results:
            return DeploymentStatus.UNKNOWN
        
        # If any component is not ready, overall status is not ready
        if any(r.status == DeploymentStatus.NOT_READY for r in results):
            return DeploymentStatus.NOT_READY
        
        # If any component needs review, overall status needs review
        if any(r.status == DeploymentStatus.NEEDS_REVIEW for r in results):
            return DeploymentStatus.NEEDS_REVIEW
        
        # If all components are ready, overall status is ready
        if all(r.status == DeploymentStatus.READY for r in results):
            return DeploymentStatus.READY
        
        return DeploymentStatus.UNKNOWN
    
    def _calculate_readiness_score(self, results: List[DeploymentValidationResult]) -> float:
        """Calculate deployment readiness score (0-100)."""
        if not results:
            return 0.0
        
        score_map = {
            DeploymentStatus.READY: 100,
            DeploymentStatus.NEEDS_REVIEW: 60,
            DeploymentStatus.NOT_READY: 0,
            DeploymentStatus.UNKNOWN: 0
        }
        
        total_score = sum(score_map[result.status] for result in results)
        return round(total_score / len(results), 2)
    
    def _identify_deployment_blockers(self, results: List[DeploymentValidationResult]) -> List[str]:
        """Identify deployment blockers."""
        blockers = []
        
        for result in results:
            if result.status == DeploymentStatus.NOT_READY:
                blockers.append(f"{result.component}: {result.message}")
        
        return blockers
    
    def _identify_deployment_warnings(self, results: List[DeploymentValidationResult]) -> List[str]:
        """Identify deployment warnings."""
        warnings = []
        
        for result in results:
            if result.status == DeploymentStatus.NEEDS_REVIEW:
                warnings.append(f"{result.component}: {result.message}")
        
        return warnings
    
    def _generate_deployment_recommendations(self, results: List[DeploymentValidationResult]) -> List[str]:
        """Generate overall deployment recommendations."""
        recommendations = []
        
        # Collect all component recommendations
        for result in results:
            recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def save_deployment_report(self, report: DeploymentReport, output_path: str = None) -> str:
        """
        Save deployment validation report to file.
        
        Args:
            report: Deployment validation report
            output_path: Output file path (optional)
            
        Returns:
            Path to saved report file
        """
        if output_path is None:
            output_path = f"deployment_report_{report.report_id}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
        
        logger.info(f"Deployment report saved: {output_file}")
        return str(output_file)
    
    def generate_kubernetes_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """
        Generate Kubernetes deployment manifests.
        
        Args:
            config: Deployment configuration
            
        Returns:
            Dictionary of manifest files (filename -> content)
        """
        manifests = {}
        
        # Deployment manifest
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'cursus-runtime-testing',
                'namespace': config.namespace,
                'labels': {
                    'app': 'cursus-runtime-testing',
                    'environment': config.environment.value
                }
            },
            'spec': {
                'replicas': config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': 'cursus-runtime-testing'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'cursus-runtime-testing'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'cursus-runtime',
                            'image': config.docker_image,
                            'ports': [{
                                'containerPort': config.port
                            }],
                            'env': [
                                {'name': k, 'value': v} 
                                for k, v in config.environment_variables.items()
                            ],
                            'resources': {
                                'limits': config.resource_limits,
                                'requests': {
                                    k: v for k, v in config.resource_limits.items()
                                    if k in ['memory', 'cpu']
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': config.health_check_endpoint,
                                    'port': config.port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': config.health_check_endpoint,
                                    'port': config.port
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'cursus-runtime-testing-service',
                'namespace': config.namespace,
                'labels': {
                    'app': 'cursus-runtime-testing'
                }
            },
            'spec': {
                'selector': {
                    'app': 'cursus-runtime-testing'
                },
                'ports': [{
                    'port': 80,
                    'targetPort': config.port,
                    'protocol': 'TCP'
                }],
                'type': 'ClusterIP'
            }
        }
        
        manifests['deployment.yaml'] = yaml.dump(deployment_manifest, default_flow_style=False)
        manifests['service.yaml'] = yaml.dump(service_manifest, default_flow_style=False)
        
        return manifests
    
    def generate_docker_compose(self, config: DeploymentConfig) -> str:
        """
        Generate Docker Compose configuration.
        
        Args:
            config: Deployment configuration
            
        Returns:
            Docker Compose YAML content
        """
        compose_config = {
            'version': '3.8',
            'services': {
                'cursus-runtime-testing': {
                    'image': config.docker_image,
                    'ports': [f"{config.port}:{config.port}"],
                    'environment': config.environment_variables,
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': [f"curl -f http://localhost:{config.port}{config.health_check_endpoint} || exit 1"],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    }
                }
            }
        }
        
        # Add resource limits if specified
        if config.resource_limits:
            deploy_config = {}
            if 'memory' in config.resource_limits:
                deploy_config['memory'] = config.resource_limits['memory']
            if 'cpu' in config.resource_limits:
                deploy_config['cpus'] = config.resource_limits['cpu']
            
            if deploy_config:
                compose_config['services']['cursus-runtime-testing']['deploy'] = {
                    'resources': {
                        'limits': deploy_config,
                        'reservations': deploy_config
                    }
                }
        
        return yaml.dump(compose_config, default_flow_style=False)
