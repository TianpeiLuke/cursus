"""
Integration tests for RegistryManager with pipeline scenarios.

Tests the integration of RegistryManager with complex pipeline scenarios including:
- Multi-pipeline isolation
- Pipeline context switching
- Pipeline cleanup scenarios
"""

import pytest
from cursus.core.deps import RegistryManager
from cursus.core.base.step_interface import StepInterface
from ..core.deps.test_helpers import reset_all_global_state


class TestRegistryManagerPipelineIntegration:
    """Test RegistryManager integration with pipeline scenarios."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state before each test
        reset_all_global_state()

        self.manager = RegistryManager()

        # Create pipeline step interfaces (unified StepInterface model).
        self.data_spec = StepInterface(
            step_type="DataStep",
            node_type="source",
            contract={},
            spec={
                "dependencies": {},
                "outputs": {
                    "raw_data": {
                        "type": "processing_output",
                        "property_path": "properties.data",
                        "data_type": "S3Uri",
                    }
                },
            },
        )

        self.training_spec = StepInterface(
            step_type="TrainingStep",
            node_type="internal",
            contract={},
            spec={
                "dependencies": {
                    "input_data": {
                        "type": "processing_output",
                        "required": True,
                    }
                },
                "outputs": {
                    "trained_model": {
                        "type": "model_artifacts",
                        "property_path": "properties.model",
                        "data_type": "S3Uri",
                    }
                },
            },
        )

    def test_multi_pipeline_isolation(self):
        """Test isolation between multiple pipeline contexts."""
        # Create training pipeline
        training_registry = self.manager.get_registry("training_pipeline")
        training_registry.register("data_step", self.data_spec)
        training_registry.register("training_step", self.training_spec)

        # Create inference pipeline
        inference_registry = self.manager.get_registry("inference_pipeline")
        inference_registry.register("inference_data", self.data_spec)

        # Verify isolation
        training_steps = training_registry.list_step_names()
        inference_steps = inference_registry.list_step_names()

        assert len(training_steps) == 2
        assert len(inference_steps) == 1

        assert "data_step" in training_steps
        assert "training_step" in training_steps
        assert "inference_data" in inference_steps

        # Verify no cross-contamination
        assert "inference_data" not in training_steps
        assert "data_step" not in inference_steps
        assert "training_step" not in inference_steps

    def test_pipeline_context_switching(self):
        """Test switching between pipeline contexts."""
        # Create multiple pipeline contexts
        contexts = ["dev", "staging", "prod"]

        for context in contexts:
            registry = self.manager.get_registry(f"{context}_pipeline")
            registry.register("data_step", self.data_spec)
            registry.register("training_step", self.training_spec)

        # Verify all contexts exist
        all_contexts = self.manager.list_contexts()
        for context in contexts:
            assert f"{context}_pipeline" in all_contexts

        # Test context switching
        for context in contexts:
            registry = self.manager.get_registry(f"{context}_pipeline")
            steps = registry.list_step_names()
            assert len(steps) == 2
            assert "data_step" in steps
            assert "training_step" in steps

    def test_pipeline_cleanup_scenarios(self):
        """Test various pipeline cleanup scenarios."""
        # Create multiple pipelines
        pipelines = ["pipeline_a", "pipeline_b", "pipeline_c"]

        for pipeline in pipelines:
            registry = self.manager.get_registry(pipeline)
            registry.register("step1", self.data_spec)
            registry.register("step2", self.training_spec)

        # Verify all created
        assert len(self.manager.list_contexts()) == 3

        # Clear specific pipeline
        result = self.manager.clear_context("pipeline_b")
        assert result is True

        remaining_contexts = self.manager.list_contexts()
        assert len(remaining_contexts) == 2
        assert "pipeline_a" in remaining_contexts
        assert "pipeline_c" in remaining_contexts
        assert "pipeline_b" not in remaining_contexts

        # Clear all remaining
        self.manager.clear_all_contexts()
        assert len(self.manager.list_contexts()) == 0


if __name__ == "__main__":
    pytest.main([__file__])
