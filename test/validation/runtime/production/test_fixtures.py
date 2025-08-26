"""
Test fixtures for production validation tests.

Provides shared test data, mock objects, and utility functions
for testing production validation components.
"""

from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock

# Import the modules for type hints
try:
    from src.cursus.validation.runtime.production.e2e_validator import (
        E2ETestScenario, E2ETestResult, E2EValidationReport
    )
    from src.cursus.validation.runtime.production.performance_optimizer import (
        PerformanceMetrics, OptimizationRecommendation, PerformanceAnalysisReport
    )
    from src.cursus.validation.runtime.production.health_checker import (
        ComponentHealthCheck, SystemHealthReport, HealthStatus
    )
    from src.cursus.validation.runtime.production.deployment_validator import (
        DeploymentConfig, DeploymentValidationResult, DeploymentReport,
        DeploymentEnvironment, DeploymentStatus
    )
except ImportError:
    # Graceful fallback for missing dependencies
    E2ETestScenario = None
    E2ETestResult = None
    E2EValidationReport = None
    PerformanceMetrics = None
    OptimizationRecommendation = None
    PerformanceAnalysisReport = None
    ComponentHealthCheck = None
    SystemHealthReport = None
    HealthStatus = None
    DeploymentConfig = None
    DeploymentValidationResult = None
    DeploymentReport = None
    DeploymentEnvironment = None
    DeploymentStatus = None


# Sample E2E Test Scenarios
SAMPLE_E2E_SCENARIOS = [
    {
        "scenario_name": "xgboost_training_pipeline",
        "pipeline_config_path": "/fake/configs/xgboost_training.yaml",
        "expected_steps": ["data_preprocessing", "feature_engineering", "model_training", "model_evaluation"],
        "data_source": "synthetic",
        "validation_rules": {
            "min_execution_time": 30.0,
            "max_memory_usage": 2048.0,
            "required_outputs": ["/fake/outputs/model.pkl", "/fake/outputs/metrics.json"]
        },
        "timeout_minutes": 45,
        "memory_limit_gb": 4.0,
        "environment_variables": {
            "PYTHONPATH": "/app",
            "AWS_REGION": "us-east-1",
            "MODEL_TYPE": "xgboost"
        },
        "tags": ["ml", "training", "xgboost"]
    },
    {
        "scenario_name": "currency_conversion_pipeline",
        "pipeline_config_path": "/fake/configs/currency_conversion.yaml",
        "expected_steps": ["data_ingestion", "currency_conversion", "data_validation"],
        "data_source": "real",
        "validation_rules": {
            "min_execution_time": 10.0,
            "max_memory_usage": 1024.0
        },
        "timeout_minutes": 20,
        "memory_limit_gb": 2.0,
        "environment_variables": {
            "PYTHONPATH": "/app",
            "CURRENCY_API_KEY": "test_key"
        },
        "tags": ["data", "conversion", "currency"]
    }
]

# Sample Performance Metrics
SAMPLE_PERFORMANCE_METRICS = [
    {
        "timestamp": datetime.now(),
        "cpu_usage_percent": 45.5,
        "memory_usage_mb": 1024.0,
        "memory_peak_mb": 1536.0,
        "disk_io_read_mb": 50.0,
        "disk_io_write_mb": 25.0,
        "execution_time_seconds": 120.0,
        "concurrent_tasks": 2,
        "memory_available_mb": 4096.0,
        "swap_usage_mb": 0.0
    },
    {
        "timestamp": datetime.now(),
        "cpu_usage_percent": 75.2,
        "memory_usage_mb": 2048.0,
        "memory_peak_mb": 2048.0,
        "disk_io_read_mb": 100.0,
        "disk_io_write_mb": 50.0,
        "execution_time_seconds": 240.0,
        "concurrent_tasks": 4,
        "memory_available_mb": 2048.0,
        "swap_usage_mb": 512.0
    }
]

# Sample Optimization Recommendations
SAMPLE_OPTIMIZATION_RECOMMENDATIONS = [
    {
        "category": "memory",
        "severity": "high",
        "description": "High memory usage detected during peak processing",
        "suggested_action": "Implement memory optimization strategies, consider data streaming",
        "estimated_improvement": "30-50% memory reduction",
        "priority_score": 90.0,
        "implementation_effort": "high"
    },
    {
        "category": "cpu",
        "severity": "medium",
        "description": "CPU usage spikes during concurrent operations",
        "suggested_action": "Consider reducing concurrent operations or optimizing CPU-intensive tasks",
        "estimated_improvement": "20-30% performance improvement",
        "priority_score": 75.0,
        "implementation_effort": "medium"
    },
    {
        "category": "io",
        "severity": "low",
        "description": "Moderate disk I/O usage observed",
        "suggested_action": "Implement I/O optimization, consider caching strategies",
        "estimated_improvement": "15-25% I/O performance improvement",
        "priority_score": 60.0,
        "implementation_effort": "low"
    }
]

# Sample Health Check Results
SAMPLE_HEALTH_CHECK_RESULTS = [
    {
        "component_name": "core_components",
        "status": "healthy",
        "message": "All core components are accessible and functional",
        "check_duration": 1.5,
        "details": {
            "pipeline_script_executor": "OK",
            "pipeline_executor": "OK",
            "real_data_tester": "OK"
        },
        "recommendations": []
    },
    {
        "component_name": "dependencies",
        "status": "healthy",
        "message": "All required dependencies are available",
        "check_duration": 2.0,
        "details": {
            "package_versions": {
                "pydantic": "1.8.0",
                "psutil": "5.8.0",
                "boto3": "1.17.0",
                "yaml": "5.4.0",
                "pandas": "1.3.0",
                "numpy": "1.21.0"
            }
        },
        "recommendations": []
    },
    {
        "component_name": "aws_access",
        "status": "warning",
        "message": "AWS credentials configured but some services have access issues",
        "check_duration": 3.0,
        "details": {
            "credentials": "configured",
            "s3_access": "accessible",
            "sagemaker_access": "error: insufficient permissions",
            "region": "us-east-1"
        },
        "recommendations": ["Review AWS IAM permissions"]
    }
]

# Sample Deployment Configurations
SAMPLE_DEPLOYMENT_CONFIGS = [
    {
        "environment": "development",
        "docker_image": "myregistry/cursus-runtime:dev",
        "resource_limits": {"memory": "1Gi", "cpu": "500m"},
        "environment_variables": {
            "PYTHONPATH": "/app",
            "LOG_LEVEL": "debug",
            "ENVIRONMENT": "development"
        },
        "health_check_endpoint": "/health",
        "port": 8080,
        "replicas": 1,
        "namespace": "development"
    },
    {
        "environment": "staging",
        "docker_image": "myregistry/cursus-runtime:v1.0.0-rc1",
        "resource_limits": {"memory": "2Gi", "cpu": "1000m"},
        "environment_variables": {
            "PYTHONPATH": "/app",
            "LOG_LEVEL": "info",
            "ENVIRONMENT": "staging",
            "AWS_REGION": "us-east-1"
        },
        "health_check_endpoint": "/health",
        "port": 8080,
        "replicas": 2,
        "namespace": "staging"
    },
    {
        "environment": "production",
        "docker_image": "myregistry/cursus-runtime:v1.0.0",
        "resource_limits": {
            "memory": "4Gi", 
            "cpu": "2000m",
            "securityContext": "nonroot"
        },
        "environment_variables": {
            "PYTHONPATH": "/app",
            "LOG_LEVEL": "info",
            "ENVIRONMENT": "production",
            "AWS_REGION": "us-east-1",
            "SECRET_KEY": "encrypted_production_secret",
            "METRICS_PORT": "9090",
            "PROMETHEUS_ENDPOINT": "/metrics",
            "BACKUP_ENABLED": "true",
            "BACKUP_SCHEDULE": "0 2 * * *",
            "PERSISTENT_STORAGE": "/data"
        },
        "health_check_endpoint": "/health",
        "port": 8080,
        "replicas": 3,
        "namespace": "production"
    }
]

# Sample Deployment Validation Results
SAMPLE_DEPLOYMENT_VALIDATION_RESULTS = [
    {
        "component": "docker_configuration",
        "status": "ready",
        "message": "Docker configuration is valid and image is available",
        "details": {
            "docker_version": "Docker version 20.10.0",
            "image_name": "myregistry/cursus-runtime:v1.0.0",
            "image_available": True
        },
        "recommendations": []
    },
    {
        "component": "kubernetes_configuration",
        "status": "ready",
        "message": "Kubernetes cluster accessible, namespace exists",
        "details": {
            "kubectl_version": "Client Version: v1.21.0",
            "cluster_accessible": True,
            "namespace": "production",
            "namespace_valid": True
        },
        "recommendations": []
    },
    {
        "component": "security_configuration",
        "status": "needs_review",
        "message": "Security validation: 1 issues found",
        "details": {
            "environment": "production",
            "issues": ["Security context not configured"]
        },
        "recommendations": ["Configure security context for container"]
    }
]


def create_mock_e2e_scenario(scenario_name: str = "test_scenario") -> Mock:
    """Create a mock E2E test scenario."""
    mock_scenario = Mock()
    mock_scenario.scenario_name = scenario_name
    mock_scenario.pipeline_config_path = f"/fake/configs/{scenario_name}.yaml"
    mock_scenario.expected_steps = ["step1", "step2", "step3"]
    mock_scenario.data_source = "synthetic"
    mock_scenario.validation_rules = {"min_execution_time": 30.0}
    mock_scenario.timeout_minutes = 30
    mock_scenario.memory_limit_gb = 4.0
    mock_scenario.environment_variables = {"PYTHONPATH": "/app"}
    mock_scenario.tags = ["test"]
    return mock_scenario


def create_mock_e2e_result(scenario_name: str = "test_scenario", success: bool = True) -> Mock:
    """Create a mock E2E test result."""
    mock_result = Mock()
    mock_result.scenario_name = scenario_name
    mock_result.success = success
    mock_result.start_time = datetime.now()
    mock_result.end_time = datetime.now()
    mock_result.total_duration = 120.0
    mock_result.peak_memory_usage = 1024.0
    mock_result.steps_executed = 3
    mock_result.steps_failed = 0 if success else 1
    mock_result.validation_results = {"test": "passed"}
    mock_result.error_details = None if success else "Test error"
    mock_result.performance_metrics = {"duration": 120.0}
    mock_result.resource_usage = {"memory_increase_mb": 512.0}
    mock_result.warnings = []
    mock_result.success_rate = 1.0 if success else 0.67
    return mock_result


def create_mock_performance_metrics(cpu_usage: float = 50.0, memory_usage: float = 1024.0) -> Mock:
    """Create mock performance metrics."""
    mock_metrics = Mock()
    mock_metrics.timestamp = datetime.now()
    mock_metrics.cpu_usage_percent = cpu_usage
    mock_metrics.memory_usage_mb = memory_usage
    mock_metrics.memory_peak_mb = memory_usage
    mock_metrics.disk_io_read_mb = 10.0
    mock_metrics.disk_io_write_mb = 5.0
    mock_metrics.execution_time_seconds = 120.0
    mock_metrics.concurrent_tasks = 1
    mock_metrics.memory_available_mb = 4096.0
    mock_metrics.swap_usage_mb = 0.0
    return mock_metrics


def create_mock_health_check_result(component: str = "test_component", 
                                   status: str = "healthy") -> Mock:
    """Create a mock health check result."""
    mock_result = Mock()
    mock_result.component_name = component
    mock_result.status = status
    mock_result.message = f"Component {component} is {status}"
    mock_result.check_duration = 1.0
    mock_result.details = {"version": "1.0.0"}
    mock_result.recommendations = []
    mock_result.timestamp = datetime.now()
    return mock_result


def create_mock_deployment_config(environment: str = "development") -> Mock:
    """Create a mock deployment configuration."""
    mock_config = Mock()
    mock_config.environment = environment
    mock_config.docker_image = f"myapp:{environment}"
    mock_config.resource_limits = {"memory": "2Gi", "cpu": "1000m"}
    mock_config.environment_variables = {
        "PYTHONPATH": "/app",
        "ENVIRONMENT": environment
    }
    mock_config.health_check_endpoint = "/health"
    mock_config.port = 8080
    mock_config.replicas = 1 if environment == "development" else 3
    mock_config.namespace = environment
    return mock_config


def create_mock_deployment_validation_result(component: str = "test_component",
                                            status: str = "ready") -> Mock:
    """Create a mock deployment validation result."""
    mock_result = Mock()
    mock_result.component = component
    mock_result.status = status
    mock_result.message = f"Component {component} is {status}"
    mock_result.details = {"test": "value"}
    mock_result.recommendations = []
    mock_result.validation_time = datetime.now()
    return mock_result


def get_sample_pipeline_config() -> Dict[str, Any]:
    """Get a sample pipeline configuration for testing."""
    return {
        "pipeline_name": "test_pipeline",
        "version": "1.0.0",
        "steps": [
            {
                "name": "data_preprocessing",
                "type": "processing",
                "script_path": "/app/scripts/preprocess.py",
                "inputs": {
                    "raw_data": "/data/input/raw_data.csv"
                },
                "outputs": {
                    "processed_data": "/data/processed/processed_data.csv"
                },
                "parameters": {
                    "batch_size": 1000,
                    "normalize": True
                }
            },
            {
                "name": "model_training",
                "type": "training",
                "script_path": "/app/scripts/train.py",
                "inputs": {
                    "training_data": "/data/processed/processed_data.csv"
                },
                "outputs": {
                    "model": "/data/models/model.pkl",
                    "metrics": "/data/metrics/training_metrics.json"
                },
                "parameters": {
                    "algorithm": "xgboost",
                    "max_depth": 6,
                    "learning_rate": 0.1
                }
            }
        ],
        "environment": {
            "python_version": "3.9",
            "requirements": [
                "pandas>=1.3.0",
                "scikit-learn>=1.0.0",
                "xgboost>=1.5.0"
            ]
        }
    }


def get_sample_execution_context() -> Dict[str, Any]:
    """Get a sample execution context for testing."""
    return {
        "workspace_dir": "/tmp/test_workspace",
        "environment_variables": {
            "PYTHONPATH": "/app",
            "AWS_REGION": "us-east-1",
            "LOG_LEVEL": "info"
        },
        "timeout_seconds": 1800,
        "memory_limit_mb": 4096,
        "execution_id": "test_execution_123",
        "timestamp": datetime.now().isoformat()
    }


def get_sample_system_metrics() -> Dict[str, Any]:
    """Get sample system metrics for testing."""
    return {
        "cpu_count": 4,
        "cpu_freq": {"current": 2400.0, "min": 800.0, "max": 3200.0},
        "boot_time": "2023-01-01T00:00:00",
        "load_average": [1.2, 1.5, 1.8],
        "network_connections": 25,
        "process_count": 150,
        "memory_total_gb": 16.0,
        "disk_total_gb": 500.0
    }


class MockProcessMemoryInfo:
    """Mock process memory info for testing."""
    
    def __init__(self, rss: int = 1024 * 1024 * 512):  # 512MB default
        self.rss = rss


class MockVirtualMemory:
    """Mock virtual memory info for testing."""
    
    def __init__(self, total: int = 16 * 1024**3, available: int = 8 * 1024**3, 
                 percent: float = 50.0, used: int = 8 * 1024**3):
        self.total = total
        self.available = available
        self.percent = percent
        self.used = used


class MockDiskUsage:
    """Mock disk usage info for testing."""
    
    def __init__(self, total: int = 500 * 1024**3, used: int = 250 * 1024**3, 
                 free: int = 250 * 1024**3):
        self.total = total
        self.used = used
        self.free = free


class MockDiskIOCounters:
    """Mock disk I/O counters for testing."""
    
    def __init__(self, read_bytes: int = 100 * 1024**2, write_bytes: int = 50 * 1024**2):
        self.read_bytes = read_bytes
        self.write_bytes = write_bytes


def create_mock_subprocess_result(returncode: int = 0, stdout: str = "", stderr: str = "") -> Mock:
    """Create a mock subprocess result."""
    mock_result = Mock()
    mock_result.returncode = returncode
    mock_result.stdout = stdout
    mock_result.stderr = stderr
    return mock_result


def create_mock_boto3_session(has_credentials: bool = True, region: str = "us-east-1") -> Mock:
    """Create a mock boto3 session."""
    mock_session = Mock()
    
    if has_credentials:
        mock_credentials = Mock()
        mock_session.get_credentials.return_value = mock_credentials
    else:
        mock_session.get_credentials.return_value = None
    
    mock_session.region_name = region
    return mock_session


def create_mock_boto3_client(service: str = "s3", accessible: bool = True) -> Mock:
    """Create a mock boto3 client."""
    mock_client = Mock()
    
    if service == "s3":
        if accessible:
            mock_client.list_buckets.return_value = {"Buckets": []}
        else:
            from botocore.exceptions import ClientError
            mock_client.list_buckets.side_effect = ClientError(
                {"Error": {"Code": "AccessDenied"}}, "ListBuckets"
            )
    elif service == "sagemaker":
        if accessible:
            mock_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}
        else:
            from botocore.exceptions import ClientError
            mock_client.list_training_jobs.side_effect = ClientError(
                {"Error": {"Code": "AccessDenied"}}, "ListTrainingJobs"
            )
    
    return mock_client


# Test data validation functions
def validate_e2e_scenario_data(scenario_data: Dict[str, Any]) -> bool:
    """Validate E2E scenario test data."""
    required_fields = ["scenario_name", "pipeline_config_path", "expected_steps"]
    return all(field in scenario_data for field in required_fields)


def validate_performance_metrics_data(metrics_data: Dict[str, Any]) -> bool:
    """Validate performance metrics test data."""
    required_fields = ["cpu_usage_percent", "memory_usage_mb", "execution_time_seconds"]
    return all(field in metrics_data for field in required_fields)


def validate_health_check_data(health_data: Dict[str, Any]) -> bool:
    """Validate health check test data."""
    required_fields = ["component_name", "status", "message"]
    return all(field in health_data for field in required_fields)


def validate_deployment_config_data(config_data: Dict[str, Any]) -> bool:
    """Validate deployment configuration test data."""
    required_fields = ["environment", "docker_image", "resource_limits"]
    return all(field in config_data for field in required_fields)
