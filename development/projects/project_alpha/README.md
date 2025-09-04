# Developer Workspace: project_alpha

This workspace was migrated from the legacy pipeline_testing structure and reorganized for better structure.

## Directory Structure

### Source Code
- `src/` - Source code implementations

### Testing & Development
- `test/` - All testing-related resources
  - `test/data/` - Test data organized by type:
    - `test/data/inputs/` - Input data for pipeline steps
    - `test/data/local_data/` - Local test data files
    - `test/data/s3_data/` - Downloaded S3 data for testing
    - `test/data/synthetic_data/` - Generated synthetic test data
  - `test/logs/` - Execution logs from test runs
  - `test/metadata/` - Pipeline execution metadata
  - `test/outputs/` - Output data from pipeline test runs
  - `test/validation_reports/` - Validation reports

## Migration Information

- Migrated from: /Users/tianpeixie/github_workspace/cursus/pipeline_testing
- Migration date: 1756792404.2966402
- Target developer: developer_1 → project_alpha
- Structure reorganized: Testing-related folders moved under `test/` directory

## Next Steps

1. Review migrated data for completeness
2. Update any hardcoded paths in scripts to use the new structure
3. Create step implementations in `src/cursus_dev/steps/` as needed
4. Test workspace functionality with the new organized structure
