# MIMS Payload Path Handling Fix

**Date:** July 12, 2025  
**Status:** COMPLETE  
**Completion:** 100%  
**Author:** Luke Xie  
**Related Documents:**
- [2025-07-07_project_status_update.md](./2025-07-07_project_status_update.md)
- [specification_driven_xgboost_pipeline_plan.md](./specification_driven_xgboost_pipeline_plan.md)

## Overview

This document outlines a critical fix implemented to resolve path handling issues in the MIMS payload step. The issue was causing pipeline failures due to conflicts between SageMaker's directory-based output handling and our script's file-based output approach.

## Issue Description

The MIMS payload step was encountering errors during execution, specifically when trying to write the payload archive:

```
ERROR:__main__:ERROR: Archive path exists but is a directory: /opt/ml/processing/output/payload.tar.gz
ERROR:__main__:Error creating payload archive: [Errno 21] Is a directory: '/opt/ml/processing/output/payload.tar.gz'
```

### Root Cause Analysis

1. **SageMaker Behavior**: SageMaker creates a directory at the path specified in `ProcessingOutput`'s `source` parameter before the script executes.

2. **Contract Configuration**: Our script contract specified the output path as `/opt/ml/processing/output/payload.tar.gz`.

3. **Script Behavior**: The script attempted to create a file at the same path where SageMaker had already created a directory.

4. **Result**: The script failed with a "Is a directory" error because it couldn't create a file where a directory already existed.

## Solution Implemented

### Contract Update

Modified the MIMS payload script contract to use a directory path instead of a file path:

```python
# Before (causing conflict)
MIMS_PAYLOAD_CONTRACT = ScriptContract(
    # Other fields...
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output/payload.tar.gz"
    },
)

# After (fixing the issue)
MIMS_PAYLOAD_CONTRACT = ScriptContract(
    # Other fields...
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output"
    },
)
```

### Builder Update

Updated the builder to generate an S3 destination path without the file suffix:

```python
# Before
destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}/payload.tar.gz"

# After
destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}"
```

### Script Behavior

The script still writes to `/opt/ml/processing/output/payload.tar.gz`, but now it's creating a file within the directory SageMaker has prepared, rather than trying to replace the directory with a file.

## Validation Process

The fix was validated through several tests:

1. **Individual Step Test**: Tested the MIMS payload step in isolation
   - Verified it successfully creates the payload archive
   - Confirmed the file is correctly written to S3

2. **Integration Test**: Tested the payload step connected to the registration step
   - Verified the registration step receives the correct S3 path
   - Confirmed the MIMS registration process validates the file successfully

3. **End-to-End Test**: Tested the complete XGBoostTrainEvaluateE2ETemplate
   - Verified all steps execute successfully
   - Confirmed model registration completes with payload validation

## MIMS Registration Validation Insight

An interesting finding from this investigation was how MIMS path validation works:

1. During pipeline definition (when connecting steps), path validation is bypassed for SageMaker property references:
   ```python
   # In MimsModelRegistrationProcessor.validate_processing_job_input_file:
   try:
       if not input_file_location.endswith(".tar.gz"):
           return False
   except AttributeError:
       # If this error is thrown than the input_file_location is a SageMaker property set in the pipeline
       if "S3" not in input_file_location.expr["Get"]:
           return False
   ```

2. This means the strict `.tar.gz` validation only applies to direct string paths, not property references from pipeline steps.

3. At runtime, the MIMS registration script focuses on the file content, not the path suffix. It processes the payload file regardless of the S3 path structure, as long as it's a valid tar.gz file.

## Benefits of the Fix

1. **Increased Reliability**: Eliminated a common source of pipeline failures
2. **Better SageMaker Integration**: Properly aligns with SageMaker's approach to processing outputs
3. **Maintained Validation**: Preserves compatibility with MIMS registration validation
4. **Simplified Contract**: Makes the contract more intuitive by specifying a directory rather than a specific filename

## Conclusion

This fix demonstrates the importance of understanding the runtime behavior of SageMaker processing steps and how they interact with script contracts. By aligning our contract design with SageMaker's directory-based approach to output handling, we've created a more robust and reliable pipeline component.

The investigation also revealed interesting insights about the MIMS validation process, which will be valuable for future pipeline development and troubleshooting.
