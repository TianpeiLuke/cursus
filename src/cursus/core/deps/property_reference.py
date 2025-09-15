"""
Property Reference module for SageMaker property path handling.

This module provides the PropertyReference class for bridging between definition-time specifications 
and runtime property references in the SageMaker pipeline context. It handles complex property paths
across various SageMaker step types.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import re

from pydantic import BaseModel, Field, field_validator, ConfigDict

from ..base import OutputSpec


class PropertyReference(BaseModel):
    """Lazy evaluation reference bridging definition-time and runtime."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    step_name: str = Field(
        ..., description="Name of the step that produces this output", min_length=1
    )
    output_spec: OutputSpec = Field(
        ..., description="Output specification for the referenced output"
    )

    @field_validator("step_name")
    @classmethod
    def validate_step_name(cls, v: str) -> str:
        """Validate step name is not empty."""
        if not v or not v.strip():
            raise ValueError("step_name cannot be empty or whitespace")
        return v.strip()

    def to_sagemaker_property(self) -> Dict[str, str]:
        """Convert to SageMaker Properties dictionary format at pipeline definition time."""
        # Remove "properties." prefix if present
        property_path = self.output_spec.property_path
        if property_path.startswith("properties."):
            property_path = property_path[11:]  # Remove "properties."

        return {"Get": f"Steps.{self.step_name}.{property_path}"}

    def to_runtime_property(self, step_instances: Dict[str, Any]) -> Any:
        """
        Create an actual SageMaker property reference using step instances.

        This method navigates the property path to create a proper SageMaker
        Properties object that can be used at runtime.

        Args:
            step_instances: Dictionary mapping step names to step instances

        Returns:
            SageMaker Properties object for the referenced property

        Raises:
            ValueError: If the step is not found or property path is invalid
            AttributeError: If any part of the property path is invalid
        """
        if self.step_name not in step_instances:
            raise ValueError(
                f"Step '{self.step_name}' not found in step instances. Available steps: {list(step_instances.keys())}"
            )

        # Get the step instance
        step_instance = step_instances[self.step_name]

        # Parse and navigate the property path
        path_parts = self._parse_property_path(self.output_spec.property_path)

        # Use helper method to navigate property path
        return self._get_property_value(step_instance.properties, path_parts)

    def _get_property_value(
        self, obj: Any, path_parts: List[Union[str, Tuple[str, Union[str, int]]]]
    ) -> Any:
        """
        Navigate through the property path to get the final value.

        Args:
            obj: The object to start navigation from
            path_parts: List of path parts from _parse_property_path

        Returns:
            The value at the end of the property path

        Raises:
            AttributeError: If any part of the path is invalid
            ValueError: If a path part has an invalid format
        """
        current_obj = obj

        # Navigate through each part of the path
        for part in path_parts:
            if isinstance(part, str):
                # Regular attribute access
                current_obj = getattr(current_obj, part)
            elif isinstance(part, tuple) and len(part) == 2:
                # Dictionary access with [key]
                attr_name, key = part
                if attr_name:  # If there's an attribute before the bracket
                    current_obj = getattr(current_obj, attr_name)
                # Handle the key access
                if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                    # Array index - convert string digits to int if needed
                    idx = key if isinstance(key, int) else int(key)
                    current_obj = current_obj[idx]
                else:  # Dictionary key
                    current_obj = current_obj[key]
            else:
                raise ValueError(f"Invalid path part: {part}")

        return current_obj

    def _parse_property_path(self, path: str) -> List[Union[str, Tuple[str, Union[str, int]]]]:
        """
        Parse a property path into a sequence of access operations.

        This method handles various SageMaker property path formats, including:
        - Regular attribute access: "properties.ModelArtifacts.S3ModelArtifacts"
        - Dictionary access: "properties.Outputs['DATA']"
        - Array indexing: "properties.TrainingJobSummaries[0]"
        - Mixed patterns: "properties.Config.Outputs['data'].Sub[0].Value"

        Args:
            path: Property path as a string

        Returns:
            List of access operations, where each operation is either:
            - A string for attribute access
            - A tuple (attr_name, key) for dictionary access or array indexing
        """
        # Remove "properties." prefix if present
        if path.startswith("properties."):
            path = path[11:]  # Remove "properties."

        result = []

        # Regular expression patterns:
        # 1. Dictionary access: Outputs['key'] or Outputs["key"]
        dict_pattern = re.compile(r'(\w+)\[([\'"]?)([^\]\'\"]+)\2\]')
        # 2. Array indexing: Array[0]
        array_pattern = re.compile(r"(\w+)\[(\d+)\]")
        # 3. Detect complex case with dot after bracket: Sub[0].Value
        complex_pattern = re.compile(r"([^.]+\[\d+\])\.(.+)")

        # Split by dots first, but preserve quoted parts and brackets
        parts = []
        current = ""
        in_brackets = False
        bracket_depth = 0

        for char in path:
            if char == "." and not in_brackets:
                if current:
                    parts.append(current)
                    current = ""
            elif char == "[":
                in_brackets = True
                bracket_depth += 1
                current += char
            elif char == "]":
                bracket_depth -= 1
                if bracket_depth == 0:
                    in_brackets = False
                current += char
            else:
                current += char

        if current:
            parts.append(current)

        # Process each part
        i = 0
        while i < len(parts):
            part = parts[i]

            # Handle the complex case: "Sub[0].Value"
            complex_match = complex_pattern.match(part)
            if complex_match:
                # Split into bracket part and property part
                bracket_part = complex_match.group(1)  # "Sub[0]"
                property_part = complex_match.group(2)  # "Value"

                # Process the bracket part
                array_match = array_pattern.match(bracket_part)
                if array_match:
                    # Add the array name
                    array_name = array_match.group(1)
                    result.append(array_name)

                    # Add the array index as a tuple with empty attr_name
                    array_index = int(array_match.group(2))
                    result.append(("", array_index))

                # Add the property part
                result.append(property_part)

                i += 1
                continue

            # Check if this part contains dictionary access
            dict_match = dict_pattern.match(part)
            if dict_match:
                # Extract the attribute name and key
                attr_name = dict_match.group(1)
                quote_type = dict_match.group(2)  # This will be ' or " or empty
                key = dict_match.group(3)

                # Handle numeric indices
                if not quote_type and key.isdigit():
                    key = int(key)

                # Add a tuple for dictionary access
                result.append((attr_name, key))
            else:
                # Check for pure array indexing
                array_match = array_pattern.match(part)
                if array_match:
                    # Add the array name
                    array_name = array_match.group(1)
                    result.append(array_name)

                    # Add the array index as a tuple with empty attr_name
                    array_index = int(array_match.group(2))
                    result.append(("", array_index))
                else:
                    # Regular attribute access
                    result.append(part)

            i += 1

        return result

    def __str__(self) -> str:
        return f"{self.step_name}.{self.output_spec.logical_name}"

    def __repr__(self) -> str:
        return f"PropertyReference(step='{self.step_name}', output='{self.output_spec.logical_name}')"
