# Developer Workspace: developer_2

## Directory Structure

### Step Implementation
- `src/cursus_dev/steps/builders/` - Step builder implementations
- `src/cursus_dev/steps/configs/` - Step configuration files
- `src/cursus_dev/steps/contracts/` - Step contract definitions
- `src/cursus_dev/steps/specs/` - Step specification files
- `src/cursus_dev/steps/scripts/` - Step script implementations

### Testing and Validation
- `test/` - Developer-specific tests
- `validation_reports/` - Validation reports and results

### Data Directories (created as needed)
- `inputs/` - Input data for pipeline steps
- `outputs/` - Output data from pipeline steps
- `local_data/` - Local test data files
- `s3_data/` - Downloaded S3 data for testing
- `synthetic_data/` - Generated synthetic test data
- `logs/` - Execution logs
- `metadata/` - Pipeline execution metadata

## Workspace Isolation Principle

Everything that happens within this developer workspace stays in this workspace.
Only code within `src/cursus/` (the shared core) is shared across all workspaces.

## Usage

1. Implement your step components in the appropriate `src/cursus_dev/steps/` subdirectories
2. Create tests in the `test/` directory
3. Use the workspace-aware runtime validation tools for testing
4. Generate validation reports in `validation_reports/`

## Integration

When ready to integrate your work:
1. Use the integration staging area: `../../integration_staging/`
2. Follow the workspace-to-production pathway
3. Coordinate with other developers through the shared integration process
