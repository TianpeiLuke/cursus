"""NVMe-aware security patching for SageMaker training steps.

MODSWorkflowHelper._patch_training_step unconditionally injects volume_kms_key
from SAIS secure_config. NVMe-backed GPU instances (ml.p4d, ml.g5, ml.g4dn,
ml.p3dn, ml.trn, ml.inf, any *d/*dn family) reject VolumeKmsKeyId because they
use hardware encryption. This module patches _patch_training_step to skip
volume_kms_key for those instances while preserving all other security settings.

Reference: OfficeHour-1553, CMLS-Model-1049
Remove once MODSWorkflowHelper ships its own NVMe gate.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install_nvme_aware_security_patch() -> bool:
    """Patch MODSWorkflowHelper._patch_training_step to be NVMe-aware.

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
                "nvme_security: skipped volume_kms_key for NVMe instance %s",
                instance_type,
            )
            return

        _orig_patch_training_step(step, secure_config)

    SagemakerPipelineHelper._patch_training_step = _patch_training_step_nvme_aware
    SagemakerPipelineHelper._nvme_aware_patched = True
    logger.info("nvme_security: installed NVMe-aware _patch_training_step override")
    return True
