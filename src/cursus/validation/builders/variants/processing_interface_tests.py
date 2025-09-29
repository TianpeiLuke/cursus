"""
Level 1 Interface Tests for Processing step builders.

These tests focus on Processing-specific interface requirements:
- Processor creation methods (_create_processor)
- Framework-specific method signatures (SKLearn vs XGBoost)
- Step creation pattern compliance (Pattern A vs Pattern B)
- Processing-specific configuration attributes
- Environment variables and job arguments methods
"""

from typing import Dict, Any, Optional
from ..interface_tests import InterfaceTests


class ProcessingInterfaceTests(InterfaceTests):
    """
    Level 1 Processing-specific interface tests.

    Extends the base InterfaceTests with Processing step-specific validations
    based on processing_step_builder_patterns.md analysis.
    """

    def __init__(
        self,
        builder_class,
        step_info: Optional[Dict[str, Any]] = None,
        config=None,
        spec=None,
        contract=None,
        step_name=None,
        verbose: bool = False,
        test_reporter=None,
        **kwargs
    ):
        """
        Initialize Processing interface tests.

        Args:
            builder_class: The Processing step builder class to test
            step_info: Processing-specific step information
            config: Optional config to use
            spec: Optional step specification
            contract: Optional script contract
            step_name: Optional step name
            verbose: Whether to print verbose output
            test_reporter: Optional function to report test results
            **kwargs: Additional arguments
        """
        # Initialize parent with new signature
        super().__init__(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose,
            test_reporter=test_reporter,
            **kwargs
        )
        
        # Store Processing-specific step info
        self.step_info = step_info or {}

    def get_step_type_specific_tests(self) -> list:
        """Return Processing-specific interface test methods."""
        return [
            "test_processor_creation_method",
            "test_step_creation_pattern_compliance",
            "test_processing_input_output_methods",
        ]

    def test_processor_creation_method(self):
        """Test that Processing builder implements _create_processor method."""
        self._assert(
            hasattr(self.builder_class, "_create_processor"),
            "Processing builder must implement _create_processor method",
        )

        # Test processor creation
        try:
            builder = self._create_builder_instance()
            processor = builder._create_processor()

            self._assert(
                processor is not None,
                "Processor creation should return a valid processor object",
            )

            # Check processor type based on framework
            framework = self.step_info.get("framework", "").lower()
            processor_class_name = processor.__class__.__name__

            if "xgboost" in framework:
                self._assert(
                    "XGBoost" in processor_class_name,
                    f"XGBoost framework should create XGBoostProcessor, got {processor_class_name}",
                )
            elif "sklearn" in framework or framework == "sklearn":
                self._assert(
                    "SKLearn" in processor_class_name,
                    f"SKLearn framework should create SKLearnProcessor, got {processor_class_name}",
                )
            else:
                # Generic processor validation
                self._assert(
                    "Processor" in processor_class_name,
                    f"Should create a Processor object, got {processor_class_name}",
                )

        except Exception as e:
            self._assert(False, f"Processor creation failed: {str(e)}")


    def test_step_creation_pattern_compliance(self):
        """Test step creation pattern compliance based on framework."""
        builder = self._create_builder_instance()

        # Check that builder has create_step method
        self._assert(
            hasattr(builder, "create_step") and callable(builder.create_step),
            "Processing builder must have create_step method",
        )

        # Check step creation pattern based on framework
        framework = self.step_info.get("framework", "").lower()
        pattern = self.step_info.get("step_creation_pattern", "Pattern A")

        if "xgboost" in framework:
            # XGBoost steps should use Pattern B (processor.run + step_args)
            self._log(
                "XGBoost framework should use Pattern B (processor.run + step_args)"
            )
            if pattern != "Pattern B":
                self._log(
                    "Warning: XGBoost step may not be using recommended Pattern B"
                )
        else:
            # SKLearn steps should use Pattern A (direct ProcessingStep creation)
            self._log(
                "SKLearn framework should use Pattern A (direct ProcessingStep creation)"
            )
            if pattern != "Pattern A":
                self._log(
                    "Warning: SKLearn step may not be using recommended Pattern A"
                )

    def test_processing_input_output_methods(self):
        """Test Processing-specific input/output methods."""
        builder = self._create_builder_instance()

        # Check input/output methods
        self._assert(
            hasattr(builder, "_get_inputs") and callable(builder._get_inputs),
            "Processing builder must have _get_inputs method",
        )

        self._assert(
            hasattr(builder, "_get_outputs") and callable(builder._get_outputs),
            "Processing builder must have _get_outputs method",
        )

        # Test method signatures for Processing-specific requirements
        import inspect

        # Check _get_inputs signature
        try:
            inputs_sig = inspect.signature(builder._get_inputs)
            inputs_params = list(inputs_sig.parameters.keys())
            self._assert(
                "inputs" in inputs_params,
                "_get_inputs method should have 'inputs' parameter",
            )
        except Exception as e:
            self._log(f"Could not inspect _get_inputs signature: {str(e)}")

        # Check _get_outputs signature
        try:
            outputs_sig = inspect.signature(builder._get_outputs)
            outputs_params = list(outputs_sig.parameters.keys())
            self._assert(
                "outputs" in outputs_params,
                "_get_outputs method should have 'outputs' parameter",
            )
        except Exception as e:
            self._log(f"Could not inspect _get_outputs signature: {str(e)}")
