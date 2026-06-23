"""NVMe-aware security patching for SageMaker training and processing steps.

MODSWorkflowHelper._patch_training_step and _patch_processing_step unconditionally
inject volume_kms_key from SAIS secure_config. NVMe-backed GPU instances (ml.p4d,
ml.g5, ml.g4dn, ml.p3dn, ml.trn, ml.inf, any *d/*dn family) reject VolumeKmsKeyId
because they use hardware encryption. This module patches both methods to skip
volume_kms_key for those instances while preserving all other security settings.

The processing step patch uses hybrid logic:
  - If processor._skip_volume_kms is explicitly set (True/False), honor it.
  - Otherwise, auto-detect from processor.instance_type via instance_supports_kms().

Reference: OfficeHour-1553, CMLS-Model-1049
Remove once MODSWorkflowHelper ships its own NVMe gate.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SENTINEL = object()


def install_nvme_aware_security_patch() -> bool:
    """Patch MODSWorkflowHelper to be NVMe-aware for both training and processing steps.

    For NVMe instances, clears volume_kms_key so it's excluded from the API request.
    Uses hybrid logic: config override (processor._skip_volume_kms) > auto-detect
    from instance_supports_kms(processor.instance_type).

    Idempotent — safe to call multiple times. Returns True if the patch was
    installed (or was already installed), False if dependencies are unavailable.
    """
    try:
        from sagemaker.utils import instance_supports_kms
        from mods_workflow_helper.sagemaker_pipeline_helper import (
            SagemakerPipelineHelper,
        )
    except ImportError:
        logger.debug(
            "nvme_security: mods_workflow_helper or sagemaker not available, skipping patch"
        )
        return False

    if getattr(SagemakerPipelineHelper, "_nvme_aware_patched", False):
        return True

    _orig_patch_training_step = SagemakerPipelineHelper._patch_training_step
    _orig_patch_processing_step = SagemakerPipelineHelper._patch_processing_step

    @staticmethod
    def _patch_training_step_nvme_aware(step, secure_config):
        estimator = step.estimator
        instance_type = getattr(estimator, "instance_type", None)

        if isinstance(instance_type, str) and not instance_supports_kms(instance_type):
            estimator.encrypt_inter_container_traffic = True
            if not getattr(estimator, "subnets", None) or not getattr(
                estimator, "security_group_ids", None
            ):
                estimator.subnets = secure_config.vpc_subnets
                estimator.security_group_ids = [secure_config.security_group]
            if hasattr(estimator, "output_kms_key") and not estimator.output_kms_key:
                estimator.output_kms_key = secure_config.kms_key
            logger.info(
                "nvme_security: skipped volume_kms_key for NVMe instance %s (training)",
                instance_type,
            )
            return

        _orig_patch_training_step(step, secure_config)

    @staticmethod
    def _patch_processing_step_nvme_aware(step, secure_config):
        processor = step.processor
        # Fallback: extract from step_args if processor is None
        if processor is None and hasattr(step, "step_args") and step.step_args:
            processor = step.step_args.func_args[0]

        if processor is None:
            _orig_patch_processing_step(step, secure_config)
            return

        # Hybrid logic: config override > auto-detect from instance type
        skip = getattr(processor, "_skip_volume_kms", _SENTINEL)
        if skip is _SENTINEL or skip is None:
            # Auto-detect from instance type
            instance_type = getattr(processor, "instance_type", None)
            skip = isinstance(instance_type, str) and not instance_supports_kms(
                instance_type
            )
        # skip is now True/False

        if skip:
            from sagemaker.network import NetworkConfig

            # Apply VPC config
            if hasattr(processor, "network_config") and not processor.network_config:
                processor.network_config = NetworkConfig(
                    enable_network_isolation=False,
                    security_group_ids=[secure_config.security_group],
                    subnets=secure_config.vpc_subnets,
                )
            # Actively clear volume_kms_key — SDK defaults may have injected it
            if hasattr(processor, "volume_kms_key"):
                processor.volume_kms_key = None
            # Apply output encryption
            if hasattr(processor, "output_kms_key") and not processor.output_kms_key:
                processor.output_kms_key = secure_config.kms_key
            logger.info(
                "nvme_security: skipped volume_kms_key for processing step %s "
                "(instance: %s)",
                step.name,
                getattr(processor, "instance_type", "unknown"),
            )
            return

        _orig_patch_processing_step(step, secure_config)

    SagemakerPipelineHelper._patch_training_step = _patch_training_step_nvme_aware
    SagemakerPipelineHelper._patch_processing_step = _patch_processing_step_nvme_aware
    SagemakerPipelineHelper._nvme_aware_patched = True
    logger.info(
        "nvme_security: installed NVMe-aware _patch_training_step and _patch_processing_step override"
    )
    return True
