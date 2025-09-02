# Developer 1 Workspace

This is an isolated developer workspace within the **Workspace-Aware System Architecture**. This workspace allows independent development of pipeline step implementations without interfering with other developers' work.

## Workspace Structure

```
developer_1/
├── README.md                    # This file
├── src/cursus_dev/             # Developer-specific implementations
│   └── steps/                  # Step implementations
│       ├── builders/           # Step builders
│       ├── configs/            # Configuration files
│       ├── contracts/          # Step contracts
│       ├── specs/              # Step specifications
│       └── scripts/            # Step scripts
├── test/                       # Isolated test environment
└── validation_reports/         # Workspace validation results
```

## Architecture Principles

### Workspace Isolation
- All development happens within this workspace
- No dependencies on other developer workspaces
- Independent testing and validation
- Isolated execution environment

### Shared Core Integration
- Extends shared core functionality from `src/cursus/`
- Uses common interfaces and abstractions
- Integrates with shared registry systems
- Follows shared validation patterns

## Development Workflow

### 1. Step Implementation

Create your step implementations in the appropriate directories:

#### Step Builders (`src/cursus_dev/steps/builders/`)
Implement step builders that extend the shared core builder patterns:
```python
from cursus.core.step_builder import BaseStepBuilder

class MyCustomStepBuilder(BaseStepBuilder):
    def build_step(self, config):
        # Your implementation here
        pass
```

#### Step Configurations (`src/cursus_dev/steps/configs/`)
Define configuration schemas for your steps:
```python
from cursus.core.config import BaseStepConfig

class MyCustomStepConfig(BaseStepConfig):
    # Your configuration fields here
    pass
```

#### Step Contracts (`src/cursus_dev/steps/contracts/`)
Define input/output contracts for your steps:
```python
from cursus.core.contracts import BaseStepContract

class MyCustomStepContract(BaseStepContract):
    # Your contract definition here
    pass
```

#### Step Specifications (`src/cursus_dev/steps/specs/`)
Define step specifications for pipeline integration:
```python
from cursus.core.specs import BaseStepSpec

class MyCustomStepSpec(BaseStepSpec):
    # Your specification here
    pass
```

#### Step Scripts (`src/cursus_dev/steps/scripts/`)
Implement the actual step execution scripts:
```python
# Your step execution logic here
```

### 2. Testing

Use the `test/` directory for workspace-specific tests:
- Unit tests for your step implementations
- Integration tests within your workspace
- Mock data and test fixtures
- Isolated test execution

### 3. Validation

Run workspace validation to ensure your implementations are correct:
- Step contract validation
- Configuration validation
- Integration compatibility checks
- Results stored in `validation_reports/`

## Best Practices

### Naming Conventions
- Use consistent prefixes for your step implementations
- Follow the shared core naming patterns
- Use descriptive names that indicate workspace ownership

### Code Organization
- Keep related functionality together
- Use clear module structure
- Document your implementations
- Follow shared coding standards

### Testing Strategy
- Write comprehensive unit tests
- Test edge cases and error conditions
- Use mock data for external dependencies
- Validate against step contracts

### Integration Preparation
- Ensure compatibility with shared core interfaces
- Test with different configuration scenarios
- Validate step contracts thoroughly
- Document any special requirements

## Integration Process

When ready to integrate your work:

1. **Workspace Validation**: Run full workspace validation
2. **Integration Staging**: Move to integration staging area
3. **Cross-Workspace Testing**: Test compatibility with other workspaces
4. **Production Integration**: Deploy to shared core if validation passes

## Workspace Commands

### Development
```bash
# Run workspace-specific tests
python -m pytest developer_workspaces/developers/developer_1/test/

# Validate workspace implementations
python -m cursus.validation.workspace validate developer_1

# Build workspace-specific pipeline
python -m cursus.pipeline build --workspace developer_1
```

### Integration
```bash
# Stage for integration
python -m cursus.workspace stage developer_1

# Run integration validation
python -m cursus.validation.integration validate developer_1

# Deploy to production
python -m cursus.workspace deploy developer_1
```

## Related Documentation

- `../README.md` - Developer workspaces overview
- `slipbox/1_design/workspace_aware_system_master_design.md` - System architecture
- `src/cursus/` - Shared core documentation
- `validation_reports/` - Your workspace validation results

## Getting Started

1. Review the shared core interfaces in `src/cursus/`
2. Implement your first step in the appropriate directories
3. Write tests for your implementation
4. Run workspace validation
5. Iterate and improve based on validation results

Remember: This workspace is your isolated development environment. Feel free to experiment, iterate, and develop your step implementations without worrying about affecting other developers' work.
