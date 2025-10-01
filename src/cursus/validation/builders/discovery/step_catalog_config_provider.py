"""
Step Catalog Configuration Provider.

This module provides a simplified configuration provider that leverages the existing
step catalog system to eliminate redundancy and provide proper configuration instances
for testing instead of primitive Mock() objects.

After comprehensive refactoring, this now uses a minimal approach focused on
architectural validation rather than perfect configuration mocking.
"""

import logging
from typing import Dict, Type, Any, Optional
from types import SimpleNamespace


class StepCatalogConfigProvider:
    """
    Simplified configuration provider that leverages existing step catalog system.
    
    This class eliminates redundancy by using the step catalog's existing
    configuration discovery capabilities directly, with zero hard-coded
    configuration data.
    
    After removing the redundant mock_factory.py, this now focuses on minimal
    configuration creation for architectural validation.
    """
    
    def __init__(self):
        """Initialize with lazy loading for performance."""
        self._step_catalog = None
        self._config_classes = None
        self.logger = logging.getLogger(__name__)
    
    @property
    def step_catalog(self):
        """Lazy-loaded step catalog instance."""
        if self._step_catalog is None:
            try:
                from ...step_catalog import StepCatalog
                self._step_catalog = StepCatalog(workspace_dirs=None)
            except ImportError as e:
                self.logger.debug(f"Step catalog not available: {e}")
                self._step_catalog = None
        return self._step_catalog
    
    @property
    def config_classes(self) -> Dict[str, Type]:
        """Lazy-loaded configuration classes from step catalog."""
        if self._config_classes is None and self.step_catalog is not None:
            try:
                self._config_classes = self.step_catalog.build_complete_config_classes()
            except Exception as e:
                self.logger.debug(f"Failed to build config classes: {e}")
                self._config_classes = {}
        return self._config_classes or {}
    
    def get_config_for_builder(self, builder_class: Type) -> Any:
        """
        Get proper configuration for builder using step catalog discovery.
        
        Args:
            builder_class: The step builder class requiring configuration
            
        Returns:
            Configuration instance (proper config class or fallback)
        """
        builder_name = builder_class.__name__
        
        try:
            # Direct step catalog integration - no redundant logic
            config_class_name = self._map_builder_to_config_class(builder_name)
            
            if config_class_name in self.config_classes:
                config_class = self.config_classes[config_class_name]
                config_instance = self._create_config_instance(config_class, builder_class)
                
                if config_instance:
                    self.logger.debug(f"✅ Step catalog config: {config_class_name} for {builder_name}")
                    return config_instance
            
            # Fallback to minimal mock for architectural validation
            return self._create_minimal_mock_config(builder_class)
            
        except Exception as e:
            self.logger.debug(f"Config creation failed for {builder_name}: {e}")
            return self._create_minimal_mock_config(builder_class)
    
    def _map_builder_to_config_class(self, builder_name: str) -> str:
        """Simple builder name to config class mapping."""
        if builder_name.endswith('StepBuilder'):
            base_name = builder_name[:-11]  # Remove 'StepBuilder'
            return f"{base_name}Config"
        return f"{builder_name}Config"
    
    def _create_config_instance(self, config_class: Type, builder_class: Type) -> Optional[Any]:
        """Create config instance using step catalog's from_base_config pattern."""
        try:
            # Use step catalog's existing base config creation
            base_config = self._get_base_config()
            if base_config is None:
                return None
            
            # Get minimal builder-specific data
            config_data = self._get_minimal_config_data(builder_class)
            
            # Use existing from_base_config pattern
            return config_class.from_base_config(base_config, **config_data)
            
        except Exception as e:
            self.logger.debug(f"Failed to create {config_class.__name__}: {e}")
            return None
    
    def _get_base_config(self) -> Optional[Any]:
        """Get base pipeline config with minimal required fields."""
        try:
            from ...core.base.config_base import BasePipelineConfig
            
            # Create minimal base config with only required fields
            base_config_data = {
                'author': 'test-author',
                'bucket': 'test-bucket',
                'role': 'arn:aws:iam::123456789012:role/MockRole',
                'region': 'NA',
                'service_name': 'test-service',
                'pipeline_version': '1.0.0',
                'model_class': 'test-model',
                'current_date': '2024-01-01',
                'framework_version': '1.0',
                'py_version': 'py39',
                'source_dir': '/tmp/mock_scripts',
                'project_root_folder': '/tmp/mock_project',  # Required field
                'pipeline_name': 'test-pipeline',
                'pipeline_s3_loc': 's3://test-bucket/pipeline',
            }
            
            return BasePipelineConfig(**base_config_data)
            
        except Exception as e:
            self.logger.debug(f"Failed to create BasePipelineConfig: {e}")
            return None
    
    def _get_minimal_config_data(self, builder_class: Type) -> Dict[str, Any]:
        """Get minimal configuration data using registry and step catalog for authoritative SageMaker type lookup."""
        
        # Step 1: Find step name from builder class using registry/step catalog
        step_name = self._find_step_name_for_builder(builder_class)
        if not step_name:
            self.logger.debug(f"No step name found for builder {builder_class.__name__}, using fallback")
            return self._get_fallback_config_data(builder_class)
        
        # Step 2: Get SageMaker type from registry (authoritative source)
        sagemaker_type = self._get_sagemaker_type_from_registry(step_name)
        if not sagemaker_type:
            self.logger.debug(f"No SageMaker type found for step {step_name}, using fallback")
            return self._get_fallback_config_data(builder_class)
        
        # Step 3: Generate config based on authoritative SageMaker type
        return self._generate_config_for_sagemaker_type(sagemaker_type, step_name)
    
    def _find_step_name_for_builder(self, builder_class: Type) -> Optional[str]:
        """Find step name for builder class using step catalog discovery."""
        builder_name = builder_class.__name__
        
        # Try step catalog first (most comprehensive)
        if self.step_catalog:
            try:
                for step_name in self.step_catalog.list_available_steps():
                    step_info = self.step_catalog.get_step_info(step_name)
                    if step_info and step_info.registry_data.get('builder_step_name') == builder_name:
                        return step_name
            except Exception as e:
                self.logger.debug(f"Step catalog lookup failed: {e}")
        
        # Fallback: Try registry directly
        try:
            from ...registry.step_names import get_builder_step_names
            builder_step_names = get_builder_step_names()
            for step_name, registered_builder_name in builder_step_names.items():
                if registered_builder_name == builder_name:
                    return step_name
        except Exception as e:
            self.logger.debug(f"Registry lookup failed: {e}")
        
        return None
    
    def _get_sagemaker_type_from_registry(self, step_name: str) -> Optional[str]:
        """Get SageMaker type from registry (authoritative source)."""
        try:
            from ...registry.step_names import get_sagemaker_step_type
            return get_sagemaker_step_type(step_name)
        except Exception as e:
            self.logger.debug(f"Registry SageMaker type lookup failed for {step_name}: {e}")
            
            # Fallback to step catalog
            if self.step_catalog:
                try:
                    step_info = self.step_catalog.get_step_info(step_name)
                    if step_info:
                        return step_info.registry_data.get('sagemaker_step_type')
                except Exception as e2:
                    self.logger.debug(f"Step catalog SageMaker type lookup failed: {e2}")
            
            return None
    
    def _generate_config_for_sagemaker_type(self, sagemaker_type: str, step_name: str) -> Dict[str, Any]:
        """Generate minimal configuration based on authoritative SageMaker type."""
        config_data = {}
        
        if sagemaker_type == "Processing":
            config_data.update({
                'job_type': 'training',
                'processing_instance_type': 'ml.m5.large',
                'processing_instance_count': 1,
                'processing_volume_size': 30,
                'processing_entry_point': 'process.py',
            })
            
            # Add step-specific fields based on step name
            if "ModelCalibration" in step_name:
                config_data.update({
                    'label_field': 'target',
                    'calibration_method': 'isotonic',
                })
        
        elif sagemaker_type == "Training":
            config_data.update({
                'training_instance_type': 'ml.m5.xlarge',
                'training_instance_count': 1,
                'training_volume_size': 30,
                'training_entry_point': 'train.py',
            })
        
        elif sagemaker_type == "Transform":
            config_data.update({
                'transform_instance_type': 'ml.m5.large',
                'transform_instance_count': 1,
                'job_type': 'training',
            })
        
        elif sagemaker_type == "CreateModel":
            config_data.update({
                'model_name': 'test-model',
                'instance_type': 'ml.m5.large',
                'entry_point': 'inference.py',
            })
        
        return config_data
    
    def _get_fallback_config_data(self, builder_class: Type) -> Dict[str, Any]:
        """Fallback configuration generation using builder name patterns."""
        builder_name = builder_class.__name__
        
        # Simple fallback based on name patterns (minimal)
        if "Processing" in builder_name or "ModelCalibration" in builder_name:
            config_data = {
                'job_type': 'training',
                'processing_instance_type': 'ml.m5.large',
                'processing_instance_count': 1,
                'processing_volume_size': 30,
                'processing_entry_point': 'process.py',
            }
            if "ModelCalibration" in builder_name:
                config_data.update({
                    'label_field': 'target',
                    'calibration_method': 'isotonic',
                })
            return config_data
        
        elif "Training" in builder_name:
            return {
                'training_instance_type': 'ml.m5.xlarge',
                'training_instance_count': 1,
                'training_volume_size': 30,
                'training_entry_point': 'train.py',
            }
        
        elif "Transform" in builder_name:
            return {
                'transform_instance_type': 'ml.m5.large',
                'transform_instance_count': 1,
                'job_type': 'training',
            }
        
        elif "CreateModel" in builder_name or "Model" in builder_name:
            return {
                'model_name': 'test-model',
                'instance_type': 'ml.m5.large',
                'entry_point': 'inference.py',
            }
        
        return {}
    
    def _create_minimal_mock_config(self, builder_class: Type) -> SimpleNamespace:
        """Create minimal mock configuration for architectural validation."""
        mock_config = SimpleNamespace()
        
        # Basic required fields for architectural validation
        mock_config.region = "NA"
        mock_config.pipeline_name = "test-pipeline"
        mock_config.pipeline_s3_loc = "s3://test-bucket/pipeline"
        
        # Add basic methods that builders expect
        mock_config.get_script_contract = lambda: None
        mock_config.get_image_uri = lambda: "test-image-uri"
        mock_config.get_script_path = lambda: "test_script.py"
        
        # Add minimal step-specific fields
        builder_name = builder_class.__name__
        
        if "Processing" in builder_name or "ModelCalibration" in builder_name:
            mock_config.job_type = "training"
            mock_config.processing_instance_type = "ml.m5.large"
            mock_config.processing_instance_count = 1
            mock_config.processing_volume_size = 30
            mock_config.processing_entry_point = "process.py"
            
            if "ModelCalibration" in builder_name:
                mock_config.label_field = "target"
                mock_config.calibration_method = "isotonic"
        
        elif "Training" in builder_name:
            mock_config.training_instance_type = "ml.m5.xlarge"
            mock_config.training_instance_count = 1
            mock_config.training_volume_size = 30
            mock_config.training_entry_point = "train.py"
        
        elif "Transform" in builder_name:
            mock_config.transform_instance_type = "ml.m5.large"
            mock_config.transform_instance_count = 1
            mock_config.job_type = "training"
        
        elif "CreateModel" in builder_name or "Model" in builder_name:
            mock_config.model_name = "test-model"
            mock_config.instance_type = "ml.m5.large"
            mock_config.entry_point = "inference.py"
        
        return mock_config
