---
tags:
  - design
  - implementation
  - security
  - sagemaker
  - pypi
  - codeartifact
keywords:
  - secure pypi
  - codeartifact
  - package installation
  - training container
  - processing container
  - ca_repository_arn
  - authentication
  - sagemaker-training-toolkit
topics:
  - secure package installation
  - sagemaker containers
  - python dependencies
  - authentication patterns
language: python
date of note: 2024-12-04
---

# Secure PyPI Installation Design

## Overview

This document describes the architecture and implementation patterns for secure Python package installation in SageMaker Training and Processing containers. It explains the different authentication mechanisms available, why certain approaches work in Training containers but not Processing containers, and provides implementation guidance for both scenarios.

## Related Documents
- **[Step Builder Guide](../0_developer_guide/step_builder.md)** - Builder patterns for SageMaker steps
- **[Script Development Guide](../0_developer_guide/script_development_guide.md)** - Guidelines for script development
- **[Script Contract](../0_developer_guide/script_contract.md)** - Contract specifications for scripts

> **💡 Security Context**: Secure PyPI installation protects against supply chain attacks and ensures packages come from trusted Amazon internal repositories. This is critical for production ML workloads handling sensitive customer data.

## Problem Statement

### Public PyPI Risks
Using public PyPI (https://pypi.org) in production environments poses several risks:
- **Supply Chain Attacks**: Malicious packages or compromised legitimate packages
- **Availability**: Public PyPI outages impact training jobs
- **Compliance**: Regulatory requirements may mandate internal package sources
- **Version Control**: Limited control over package versions and security patches

### Solution: Amazon CodeArtifact
Amazon uses **CodeArtifact** as a secure, internal PyPI repository:
- Curated package repository with security scanning
- Internal Amazon package hosting
- Consistent package availability
- Compliance with Amazon security standards

## Architecture Overview

### Container Types and Authentication

```mermaid
flowchart TD
    subgraph "🎯 SageMaker Training Container"
        TrainContainer["Training Job Container<br/>(sagemaker-training-toolkit)"]
        TrainToolkit["SageMaker Training Toolkit"]
        CARepArn["CA_REPOSITORY_ARN<br/>Environment Variable"]
        AutoAuth["Automatic Authentication<br/>via Toolkit"]
        
        TrainContainer --> TrainToolkit
        TrainToolkit --> CARepArn
        CARepArn --> AutoAuth
        AutoAuth --> SecurePyPI
    end
    
    subgraph "📦 SageMaker Processing Container"
        ProcContainer["Processing Job Container<br/>(sagemaker-scikit-learn, etc.)"]
        NoToolkit["❌ No Training Toolkit<br/>❌ No CA_REPOSITORY_ARN Support"]
        ManualAuth["Manual Authentication Required"]
        EnvVar["USE_SECURE_PYPI<br/>Environment Variable"]
        ScriptAuth["Script-based Authentication"]
        
        ProcContainer --> NoToolkit
        NoToolkit --> ManualAuth
        ManualAuth --> EnvVar
        EnvVar --> ScriptAuth
        ScriptAuth --> SecurePyPI
    end
    
    subgraph "🔐 Amazon CodeArtifact"
        SecurePyPI["Secure PyPI Repository<br/>arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi"]
        TokenAuth["Token-based Authentication"]
        STSRole["STS Role Assumption:<br/>SecurePyPIReadRole_{AccountId}"]
        CAClient["CodeArtifact Client"]
        
        SecurePyPI --> TokenAuth
        TokenAuth --> STSRole
        STSRole --> CAClient
    end
    
    AutoAuth -.->|Automatic| TokenAuth
    ScriptAuth -.->|Manual| TokenAuth
    
    style TrainContainer fill:#d4f1d4
    style ProcContainer fill:#f1d4d4
    style SecurePyPI fill:#d4e5f1
    style CARepArn fill:#fff4d4
    style NoToolkit fill:#ffd4d4
```

## Training Container Implementation

### CA_REPOSITORY_ARN: The Training Container Advantage

Training containers built with `sagemaker-training-toolkit` have **built-in support** for CodeArtifact authentication via the `CA_REPOSITORY_ARN` environment variable.

#### How It Works

1. **Environment Variable Configuration**
   ```python
   # In builder (e.g., builder_pytorch_training_step.py)
   env_vars = {
       "CA_REPOSITORY_ARN": "arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi"
   }
   ```

2. **Automatic Authentication by Training Toolkit**
   - The `sagemaker-training-toolkit` detects `CA_REPOSITORY_ARN`
   - Automatically configures pip to use CodeArtifact
   - Handles token retrieval and authentication
   - No script-level code required

3. **Package Installation**
   ```bash
   # Inside training script - pip automatically uses secure PyPI
   pip install pandas numpy scikit-learn
   ```

#### Implementation in Configs

```python
# src/cursus/steps/configs/config_pytorch_training_step.py
class PyTorchTrainingConfig(BasePipelineConfig):
    """Configuration for PyTorch training with secure PyPI support."""
    
    ca_repository_arn: str = Field(
        default="arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi",
        description="CodeArtifact repository ARN for secure PyPI access. "
                   "Only used when use_secure_pypi=True."
    )
    
    use_secure_pypi: bool = Field(
        default=True,
        description="Controls PyPI source for package installation. "
                   "If True (default), uses secure CodeArtifact PyPI. "
                   "If False, uses public PyPI."
    )
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the training script."""
        env_vars = super().get_environment_variables() if hasattr(super(), 'get_environment_variables') else {}
        
        # Add training-specific environment variables
        env_vars["USE_SECURE_PYPI"] = str(self.use_secure_pypi).lower()
        
        return env_vars
```

#### Implementation in Builders

```python
# src/cursus/steps/builders/builder_pytorch_training_step.py
class PyTorchTrainingStepBuilder(StepBuilderBase):
    """Builder for PyTorch Training Steps with secure PyPI support."""
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the training job.
        
        Training containers (unlike Processing) support CA_REPOSITORY_ARN.
        """
        # Delegate to config's method which handles all training-specific variables
        if hasattr(self.config, 'get_environment_variables'):
            env_vars = self.config.get_environment_variables()
        else:
            env_vars = super()._get_environment_variables()
        
        # Training containers support CA_REPOSITORY_ARN
        # Add it if use_secure_pypi is enabled
        if self.config.use_secure_pypi:
            env_vars["CA_REPOSITORY_ARN"] = self.config.ca_repository_arn
            self.log_info("Added CA_REPOSITORY_ARN to environment variables for Training container")
        
        return env_vars
```

### Why Training Containers Support CA_REPOSITORY_ARN

The `sagemaker-training-toolkit` provides infrastructure for:
- **Automatic Token Management**: Handles STS role assumption and token refresh
- **Pip Configuration**: Automatically configures pip's index-url
- **Security**: Tokens are never exposed to user code
- **Consistency**: Same behavior across all training frameworks (PyTorch, XGBoost, etc.)

### Training Container Advantages

✅ **Automatic Authentication**: No script-level code needed  
✅ **Built-in Support**: Part of training toolkit  
✅ **Clean Separation**: Config owns policy, toolkit handles implementation  
✅ **Framework Agnostic**: Works for PyTorch, XGBoost, LightGBM, etc.  
✅ **Secure**: Credentials never exposed to user scripts  

## Processing Container Implementation

### Why Processing Containers Don't Support CA_REPOSITORY_ARN

Processing containers (built with framework-specific images like `sagemaker-scikit-learn`) **do not include** the `sagemaker-training-toolkit`, which means:

❌ **No Training Toolkit**: Processing containers lack the toolkit that interprets `CA_REPOSITORY_ARN`  
❌ **No Automatic Auth**: Cannot automatically configure pip for CodeArtifact  
❌ **Environment Variable Ignored**: Setting `CA_REPOSITORY_ARN` has no effect  
❌ **Framework Specific**: Each framework image has different capabilities  

### Processing Container Solution: Manual Authentication

Since Processing containers lack automatic authentication, scripts must handle CodeArtifact authentication manually.

#### Implementation Pattern

```python
# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"

import boto3
from subprocess import check_call
import sys


def _get_secure_pypi_access_token() -> str:
    """
    Get CodeArtifact access token for secure PyPI.
    
    Authentication Flow:
    1. Get current AWS caller identity via STS
    2. Assume SecurePyPIReadRole in current account
    3. Use assumed role to get CodeArtifact authorization token
    4. Return token for pip index-url configuration
    
    Returns:
        str: Authorization token for CodeArtifact
        
    Raises:
        Exception: If token retrieval fails
    """
    try:
        # Step 1: Configure STS for regional endpoints (best practice)
        os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
        sts = boto3.client("sts", region_name="us-east-1")
        
        # Step 2: Get current account ID
        caller_identity = sts.get_caller_identity()
        account_id = caller_identity["Account"]
        
        # Step 3: Assume SecurePyPIReadRole in current account
        role_arn = f"arn:aws:iam::{account_id}:role/SecurePyPIReadRole_675292366480"
        assumed_role_object = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="SecurePypiReadRole"
        )
        credentials = assumed_role_object["Credentials"]
        
        # Step 4: Create CodeArtifact client with assumed role credentials
        code_artifact_client = boto3.client(
            "codeartifact",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name="us-west-2"
        )
        
        # Step 5: Get authorization token for CodeArtifact domain
        token = code_artifact_client.get_authorization_token(
            domain="amazon",
            domainOwner="149122183214"
        )["authorizationToken"]
        
        print("✓ Successfully retrieved secure PyPI access token")
        return token
        
    except Exception as e:
        print(f"✗ Failed to retrieve secure PyPI access token: {e}")
        raise


def install_packages_from_public_pypi(packages: list) -> None:
    """Install packages from standard public PyPI."""
    print(f"Installing {len(packages)} packages from public PyPI")
    
    try:
        check_call([sys.executable, "-m", "pip", "install", *packages])
        print("✓ Successfully installed packages from public PyPI")
    except Exception as e:
        print(f"✗ Failed to install packages from public PyPI: {e}")
        raise


def install_packages_from_secure_pypi(packages: list) -> None:
    """Install packages from secure CodeArtifact PyPI."""
    print(f"Installing {len(packages)} packages from secure PyPI")
    
    try:
        # Get authentication token
        token = _get_secure_pypi_access_token()
        
        # Construct authenticated index URL
        index_url = (
            f"https://aws:{token}@amazon-149122183214.d.codeartifact."
            f"us-west-2.amazonaws.com/pypi/secure-pypi/simple/"
        )
        
        # Install with authenticated index
        check_call([
            sys.executable, "-m", "pip", "install",
            "--index-url", index_url,
            *packages
        ])
        
        print("✓ Successfully installed packages from secure PyPI")
    except Exception as e:
        print(f"✗ Failed to install packages from secure PyPI: {e}")
        raise


def install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI) -> None:
    """
    Install packages from PyPI source based on configuration.
    
    Args:
        packages: List of package specifications
        use_secure: If True, use secure CodeArtifact PyPI
        
    Environment Variables:
        USE_SECURE_PYPI: Set to "true" to use secure PyPI
    """
    print("=" * 70)
    print("PACKAGE INSTALLATION")
    print("=" * 70)
    print(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
    print(f"Number of packages: {len(packages)}")
    print("=" * 70)
    
    try:
        if use_secure:
            install_packages_from_secure_pypi(packages)
        else:
            install_packages_from_public_pypi(packages)
            
        print("=" * 70)
        print("✓ PACKAGE INSTALLATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print("=" * 70)
        print("✗ PACKAGE INSTALLATION FAILED")
        print("=" * 70)
        raise
```

#### Usage in Processing Scripts

```python
# Load packages from requirements file
requirements_file = os.path.join(os.path.dirname(__file__), "requirements-secure.txt")

try:
    with open(requirements_file, "r") as f:
        required_packages = [
            line.strip() for line in f 
            if line.strip() and not line.strip().startswith("#")
        ]
    
    # Install packages using unified installation function
    install_packages(required_packages)
    
except FileNotFoundError:
    print(f"Warning: {requirements_file} not found")
except Exception as e:
    print(f"Error loading or installing packages: {e}")
    raise

# Now import installed packages
import pandas as pd
import numpy as np
# ... etc
```

### Processing Container Implementation in Configs

```python
# src/cursus/steps/configs/config_pytorch_model_eval_step.py
class PyTorchModelEvalConfig(BasePipelineConfig):
    """Configuration for PyTorch model evaluation (Processing container)."""
    
    # NOTE: No ca_repository_arn field - Processing containers don't support it!
    
    use_secure_pypi: bool = Field(
        default=True,
        description="Controls PyPI source for package installation. "
                   "If True (default), uses secure CodeArtifact PyPI. "
                   "If False, uses public PyPI. "
                   "Processing containers handle authentication manually in scripts."
    )
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processing script."""
        env_vars = super().get_environment_variables() if hasattr(super(), 'get_environment_variables') else {}
        
        # Processing scripts use USE_SECURE_PYPI for manual authentication
        env_vars["USE_SECURE_PYPI"] = str(self.use_secure_pypi).lower()
        
        return env_vars
```

### Processing Container Implementation in Builders

```python
# src/cursus/steps/builders/builder_pytorch_model_eval_step.py
class PyTorchModelEvalStepBuilder(StepBuilderBase):
    """Builder for PyTorch Model Evaluation Steps (Processing container)."""
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processing job.
        
        Processing containers DON'T support CA_REPOSITORY_ARN.
        Scripts must handle authentication manually via USE_SECURE_PYPI.
        """
        # Delegate to config's method
        if hasattr(self.config, 'get_environment_variables'):
            env_vars = self.config.get_environment_variables()
        else:
            env_vars = super()._get_environment_variables()
        
        # NOTE: DO NOT add CA_REPOSITORY_ARN here!
        # Processing containers don't have sagemaker-training-toolkit
        # and will ignore this environment variable
        
        self.log_info("Processing environment variables: %s", env_vars)
        return env_vars
```

### Why Manual Authentication is Required

Processing containers require manual authentication because:

1. **No Training Toolkit**: Processing containers are built from framework images (scikit-learn, pandas, etc.) without the training toolkit
2. **Different Build Process**: Processing images focus on running arbitrary code, not training-specific workflows
3. **Script Responsibility**: Scripts must handle all setup, including authentication
4. **Token Management**: Scripts must manually:
   - Assume IAM roles
   - Retrieve CodeArtifact tokens
   - Configure pip with authenticated index URL

## Architecture Comparison

### Training Containers vs Processing Containers

| Feature | Training Container | Processing Container |
|---------|-------------------|---------------------|
| **Base Image** | sagemaker-training-toolkit | Framework-specific (scikit-learn, etc.) |
| **CA_REPOSITORY_ARN Support** | ✅ Yes | ❌ No |
| **Authentication** | Automatic via toolkit | Manual via script |
| **Config Field** | `ca_repository_arn` | N/A (not used) |
| **Script Code** | Not required | Required |
| **Environment Variable** | `CA_REPOSITORY_ARN` | `USE_SECURE_PYPI` |
| **Token Management** | Automatic | Manual (boto3 + STS) |
| **Use Cases** | Training jobs | Eval, Inference, Calibration, etc. |

## Configuration Ownership Pattern

### Single Source of Truth: Configs

All environment variable logic lives in **config classes**, not builders:

```python
# ✅ CORRECT: Config owns environment variable logic
class TrainingConfig(BasePipelineConfig):
    def get_environment_variables(self) -> Dict[str, str]:
        """Config determines what environment variables to set."""
        env_vars = {}
        env_vars["USE_SECURE_PYPI"] = str(self.use_secure_pypi).lower()
        # ... other variables
        return env_vars

# ✅ CORRECT: Builder delegates to config, adds container-specific variables
class TrainingStepBuilder(StepBuilderBase):
    def _get_environment_variables(self) -> Dict[str, str]:
        """Builder delegates to config, adds CA_REPOSITORY_ARN for Training."""
        env_vars = self.config.get_environment_variables()
        
        # Training containers support CA_REPOSITORY_ARN
        if self.config.use_secure_pypi:
            env_vars["CA_REPOSITORY_ARN"] = self.config.ca_repository_arn
        
        return env_vars
```

### Delegation Pattern Benefits

✅ **Single Source of Truth**: Configs own all policy decisions  
✅ **Clear Separation**: Builders add container-specific implementation  
✅ **Maintainability**: Environment variable changes only need config updates  
✅ **Testability**: Config logic can be tested independently  
✅ **Consistency**: Same pattern across all frameworks  

## Security Considerations

### IAM Role Architecture

```mermaid
flowchart LR
    subgraph "SageMaker Container"
        Container["Training/Processing<br/>Container"]
        ExecutionRole["SageMaker<br/>Execution Role"]
    end
    
    subgraph "IAM Role Assumption"
        AssumeRole["sts:AssumeRole"]
        SecureRole["SecurePyPIReadRole<br/>(per-account)"]
    end
    
    subgraph "CodeArtifact"
        CA["CodeArtifact<br/>Domain: amazon<br/>Owner: 149122183214"]
        GetToken["GetAuthorizationToken"]
    end
    
    Container --> ExecutionRole
    ExecutionRole --> AssumeRole
    AssumeRole --> SecureRole
    SecureRole --> GetToken
    GetToken --> CA
```

### Required IAM Permissions

#### SageMaker Execution Role
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "arn:aws:iam::*:role/SecurePyPIReadRole_*"
    }
  ]
}
```

#### SecurePyPIReadRole (Target Role)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "codeartifact:GetAuthorizationToken",
        "codeartifact:ReadFromRepository"
      ],
      "Resource": [
        "arn:aws:codeartifact:us-west-2:149122183214:domain/amazon",
        "arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi"
      ]
    },
    {
      "Effect": "Allow",
      "Action": "sts:GetServiceBearerToken",
      "Resource": "*"
    }
  ]
}
```

### Security Best Practices

1. **Least Privilege**: Roles only have permissions needed for CodeArtifact access
2. **Token Expiration**: Tokens expire after 12 hours (CodeArtifact default)
3. **Regional Endpoints**: Use regional STS endpoints for better reliability
4. **Account Isolation**: Each account has its own SecurePyPIReadRole
5. **Audit Trail**: All package installations are logged in CloudTrail

## Implementation Guidelines

### When to Use Secure PyPI

✅ **Production Workloads**: Always use secure PyPI for production  
✅ **Customer Data**: Required when handling sensitive data  
✅ **Compliance**: Mandatory for regulated industries  
✅ **Internal Packages**: Only available via secure PyPI  

❌ **Development/Testing**: Public PyPI may be acceptable  
❌ **Public Demos**: When sharing code externally  

### Configuration Checklist

#### For Training Steps
- [ ] Config has `ca_repository_arn` field
- [ ] Config has `use_secure_pypi` boolean field
- [ ] Config implements `get_environment_variables()` method
- [ ] Builder delegates to config's `get_environment_variables()`
- [ ] Builder adds `CA_REPOSITORY_ARN` when `use_secure_pypi=True`
- [ ] Training script doesn't need special auth code

#### For Processing Steps
- [ ] Config does NOT have `ca_repository_arn` field
- [ ] Config has `use_secure_pypi` boolean field
- [ ] Config implements `get_environment_variables()` method
- [ ] Builder delegates to config's `get_environment_variables()`
- [ ] Builder does NOT add `CA_REPOSITORY_ARN`
- [ ] Processing script implements manual authentication functions
- [ ] Script reads `USE_SECURE_PYPI` environment variable
- [ ] Script has `install_packages()` function with conditional logic

## Common Pitfalls

### ❌ Anti-Pattern: Adding CA_REPOSITORY_ARN to Processing Containers
```python
# DON'T DO THIS - Processing containers ignore CA_REPOSITORY_ARN
class ProcessingStepBuilder:
    def _get_environment_variables(self):
        env_vars = {}
        # ❌ WRONG: Processing containers don't support this
        env_vars["CA_REPOSITORY_ARN"] = self.config.ca_repository_arn
        return env_vars
```

### ❌ Anti-Pattern: Duplicating Logic in Builder
```python
# DON'T DO THIS - Config should own this logic
class TrainingStepBuilder:
    def _get_environment_variables(self):
        env_vars = {}
        # ❌ WRONG: Logic should be in config, not builder
        env_vars["USE_SECURE_PYPI"] = str(self.config.use_secure_pypi).lower()
        env_vars["USE_PRECOMPUTED_IMPUTATION"] = "true"
        # ... more logic duplicated from config
        return env_vars
```

### ❌ Anti-Pattern: Missing Manual Auth in Processing Scripts
```python
# DON'T DO THIS - Processing scripts need authentication code
# ❌ WRONG: This will fail if USE_SECURE_PYPI=true
import pandas as pd  # Might not be installed!
import numpy as np   # Might not be installed!
```

### ✅ Correct Pattern: Delegation and Separation
```python
# ✅ CORRECT: Config owns logic, builder adds container-specific handling
class TrainingConfig:
    def get_environment_variables(self):
        return {"USE_SECURE_PYPI": str(self.use_secure_pypi).lower()}

class TrainingStepBuilder:
    def _get_environment_variables(self):
        env_vars = self.config.get_environment_variables()
        if self.config.use_secure_pypi:
            env_vars["CA_REPOSITORY_ARN"] = self.config.ca_repository_arn
        return env_vars
```

## Testing and Validation

### Testing Secure PyPI Configuration

#### Unit Tests for Config
```python
def test_training_config_environment_variables():
    """Test training config generates correct environment variables."""
    config = PyTorchTrainingConfig(
        use_secure_pypi=True,
        # ... other required fields
    )
    
    env_vars = config.get_environment_variables()
    
    assert "USE_SECURE_PYPI" in env_vars
    assert env_vars["USE_SECURE_PYPI"] == "true"
```

#### Integration Tests for Builder
```python
def test_training_builder_adds_ca_repository_arn():
    """Test training builder adds CA_REPOSITORY_ARN."""
    config = PyTorchTrainingConfig(use_secure_pypi=True)
    builder = PyTorchTrainingStepBuilder(config)
    
    env_vars = builder._get_environment_variables()
    
    assert "CA_REPOSITORY_ARN" in env_vars
    assert env_vars["CA_REPOSITORY_ARN"].startswith("arn:aws:codeartifact")
```

#### End-to-End Tests
```python
def test_processing_script_authentication():
    """Test processing script can authenticate with secure PyPI."""
    os.environ["USE_SECURE_PYPI"] = "true"
    
    # This should successfully authenticate and install
    install_packages(["pandas==1.5.0"])
    
    # Verify package is installed
    import pandas
    assert pandas.__version__ == "1.5.0"
```

## Migration Guide

### Migrating from Public PyPI to Secure PyPI

#### Step 1: Update Configs
```python
# Before
class MyTrainingConfig(BasePipelineConfig):
    pass  # No secure PyPI support

# After
class MyTrainingConfig(BasePipelineConfig):
    use_secure_pypi: bool = Field(default=True)
    ca_repository_arn: str = Field(default="arn:aws:codeartifact:...")
    
    def get_environment_variables(self) -> Dict[str, str]:
        env_vars = super().get_environment_variables() if hasattr(super(), 'get_environment_variables') else {}
        env_vars["USE_SECURE_PYPI"] = str(self.use_secure_pypi).lower()
        return env_vars
```

#### Step 2: Update Builders (Training Only)
```python
# Before
class MyTrainingStepBuilder(StepBuilderBase):
    def _get_environment_variables(self):
        return {}

# After
class MyTrainingStepBuilder(StepBuilderBase):
    def _get_environment_variables(self):
        env_vars = self.config.get_environment_variables()
        if self.config.use_secure_pypi:
            env_vars["CA_REPOSITORY_ARN"] = self.config.ca_repository_arn
        return env_vars
```

#### Step 3: Update Processing Scripts
```python
# Add to beginning of processing script (pytorch_model_eval.py example)
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"

# Add authentication functions (see full implementation above)
def _get_secure_pypi_access_token(): ...
def install_packages_from_secure_pypi(): ...
def install_packages(): ...

# Install packages before imports
install_packages(["pandas", "numpy", "scikit-learn"])

# Now import
import pandas as pd
```

## Conclusion

Secure PyPI installation in SageMaker requires different approaches for Training and Processing containers:

### Training Containers
- **Simple Configuration**: Set `CA_REPOSITORY_ARN` environment variable
- **Automatic Handling**: `sagemaker-training-toolkit` handles authentication
- **No Script Changes**: Works transparently with pip
- **Best for**: Training jobs (PyTorch, XGBoost, LightGBM, etc.)

### Processing Containers
- **Manual Authentication**: Scripts must implement authentication logic
- **Script Responsibility**: Handle STS role assumption and token retrieval
- **Explicit Control**: Full control over authentication flow
- **Best for**: Evaluation, inference, calibration, and other processing jobs

### Key Takeaways

1. **Know Your Container**: Understand which type you're using (Training with toolkit vs Processing without)
2. **Config Owns Logic**: All environment variable logic lives in config classes, not builders
3. **Builder Adds Context**: Builders delegate to config and add container-specific variables only
4. **Manual Auth Required**: Processing scripts must implement full authentication flow
5. **Security First**: Always use secure PyPI for production workloads handling sensitive data

## Future Considerations

### Potential Improvements

1. **Unified Authentication Library**: Create a shared Python module for manual authentication that can be imported by all Processing scripts instead of duplicating code
2. **Docker Image Enhancement**: Explore adding minimal authentication support to Processing container base images
3. **Configuration Validation**: Add validation to prevent misconfiguration (e.g., setting `ca_repository_arn` on Processing steps)
4. **Monitoring**: Add metrics for package installation success/failure rates across container types

### Alternative Approaches

While the current patterns (automatic for Training, manual for Processing) work well, future alternatives could include:

1. **Custom Processing Images**: Build custom Processing images with authentication support
2. **Init Containers**: Use init containers to handle authentication before main script runs
3. **Lambda Layers**: Package authentication logic as reusable layers
4. **SageMaker Extensions**: Propose extension to SageMaker Processing SDK for built-in CodeArtifact support

## References

### Internal Documentation
- **CodeArtifact Documentation**: Internal wiki for Amazon CodeArtifact usage
- **SageMaker Training Toolkit**: GitHub repository and documentation
- **SecurePyPI Role Setup**: IAM role configuration guide

### External Resources
- **AWS CodeArtifact**: https://docs.aws.amazon.com/codeartifact/
- **SageMaker Python SDK**: https://sagemaker.readthedocs.io/
- **SageMaker Training Toolkit**: https://github.com/aws/sagemaker-training-toolkit

### Related Slipbox Documents
- **[Step Builder Guide](../0_developer_guide/step_builder.md)**: Comprehensive guide to building SageMaker steps
- **[Script Development Guide](../0_developer_guide/script_development_guide.md)**: Best practices for script development
- **[Script Contract](../0_developer_guide/script_contract.md)**: Contract specifications and I/O channels
- **[Config Field Manager Guide](../0_developer_guide/config_field_manager_guide.md)**: Managing configuration fields
- **[Three Tier Config Design](../0_developer_guide/three_tier_config_design.md)**: Configuration architecture patterns

## Appendix: Complete Code Examples

### A. Training Step Complete Implementation

```python
# Config: src/cursus/steps/configs/config_pytorch_training_step.py
class PyTorchTrainingConfig(BasePipelineConfig):
    """PyTorch Training configuration with secure PyPI support."""
    
    # Tier 2: System Inputs with Defaults
    ca_repository_arn: str = Field(
        default="arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi",
        description="CodeArtifact repository ARN for secure PyPI access."
    )
    
    use_secure_pypi: bool = Field(
        default=True,
        description="Controls PyPI source for package installation."
    )
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for training script."""
        env_vars = super().get_environment_variables() if hasattr(super(), 'get_environment_variables') else {}
        env_vars["USE_SECURE_PYPI"] = str(self.use_secure_pypi).lower()
        return env_vars


# Builder: src/cursus/steps/builders/builder_pytorch_training_step.py
class PyTorchTrainingStepBuilder(StepBuilderBase):
    """PyTorch Training Step Builder."""
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for training job."""
        # Delegate to config
        env_vars = self.config.get_environment_variables()
        
        # Add Training container-specific variable
        if self.config.use_secure_pypi:
            env_vars["CA_REPOSITORY_ARN"] = self.config.ca_repository_arn
        
        return env_vars
```

### B. Processing Step Complete Implementation

```python
# Config: src/cursus/steps/configs/config_pytorch_model_eval_step.py
class PyTorchModelEvalConfig(BasePipelineConfig):
    """PyTorch Model Evaluation configuration."""
    
    # NOTE: No ca_repository_arn field!
    
    use_secure_pypi: bool = Field(
        default=True,
        description="Controls PyPI source. Processing containers handle manually."
    )
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for processing script."""
        env_vars = super().get_environment_variables() if hasattr(super(), 'get_environment_variables') else {}
        env_vars["USE_SECURE_PYPI"] = str(self.use_secure_pypi).lower()
        return env_vars


# Builder: src/cursus/steps/builders/builder_pytorch_model_eval_step.py
class PyTorchModelEvalStepBuilder(StepBuilderBase):
    """PyTorch Model Evaluation Step Builder."""
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for processing job."""
        # Delegate to config (no CA_REPOSITORY_ARN!)
        return self.config.get_environment_variables()


# Script: src/cursus/steps/scripts/pytorch_model_eval.py (beginning)
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"

def _get_secure_pypi_access_token() -> str:
    """Get CodeArtifact token via STS role assumption."""
    os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
    sts = boto3.client("sts", region_name="us-east-1")
    caller_identity = sts.get_caller_identity()
    
    assumed_role = sts.assume_role(
        RoleArn=f"arn:aws:iam::{caller_identity['Account']}:role/SecurePyPIReadRole_675292366480",
        RoleSessionName="SecurePypiReadRole"
    )
    
    ca_client = boto3.client(
        "codeartifact",
        aws_access_key_id=assumed_role["Credentials"]["AccessKeyId"],
        aws_secret_access_key=assumed_role["Credentials"]["SecretAccessKey"],
        aws_session_token=assumed_role["Credentials"]["SessionToken"],
        region_name="us-west-2"
    )
    
    return ca_client.get_authorization_token(
        domain="amazon",
        domainOwner="149122183214"
    )["authorizationToken"]

def install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI):
    """Install packages from appropriate PyPI source."""
    if use_secure:
        token = _get_secure_pypi_access_token()
        index_url = f"https://aws:{token}@amazon-149122183214.d.codeartifact.us-west-2.amazonaws.com/pypi/secure-pypi/simple/"
        check_call([sys.executable, "-m", "pip", "install", "--index-url", index_url, *packages])
    else:
        check_call([sys.executable, "-m", "pip", "install", *packages])

# Install before imports
install_packages(["pandas", "numpy", "scikit-learn"])
import pandas as pd
import numpy as np
```

## Summary

This document provides a comprehensive guide to secure PyPI installation in SageMaker containers. The key distinction between Training and Processing containers dictates the implementation approach:

- **Training containers** benefit from `sagemaker-training-toolkit` providing automatic authentication via `CA_REPOSITORY_ARN`
- **Processing containers** require manual authentication implementation in scripts using `USE_SECURE_PYPI`

Both approaches achieve the same security goal of using Amazon's internal CodeArtifact repository, but the implementation complexity differs based on container capabilities. Following the patterns documented here ensures consistent, secure, and maintainable package installation across all SageMaker step types.
