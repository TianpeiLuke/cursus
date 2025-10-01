"""
Step type color coding system from legacy report generators.

This module provides color coding for different SageMaker step types to maintain
visual consistency with legacy report generation scripts.
"""

from typing import Dict


class StepTypeColorScheme:
    """Step type color coding system from legacy report generators."""
    
    STEP_TYPE_COLORS = {
        "Training": "#FF6B6B",      # Red
        "Transform": "#4ECDC4",     # Teal
        "CreateModel": "#45B7D1",   # Blue
        "Processing": "#96CEB4",    # Green
        "Base": "#9B59B6",          # Purple
        "Utility": "#F39C12",       # Orange
        "Lambda": "#E74C3C",        # Dark Red
        "RegisterModel": "#2ECC71", # Emerald
    }
    
    @classmethod
    def get_color_for_step_type(cls, step_type: str) -> str:
        """Get color for step type with fallback."""
        return cls.STEP_TYPE_COLORS.get(step_type, "#95A5A6")  # Gray fallback
    
    @classmethod
    def get_color_for_builder(cls, canonical_name: str, step_catalog) -> str:
        """Get color for builder based on its step type."""
        try:
            step_info = step_catalog.get_step_info(canonical_name)
            if step_info:
                step_type = step_info.registry_data.get('sagemaker_step_type')
                return cls.get_color_for_step_type(step_type)
            return cls.STEP_TYPE_COLORS.get("Base", "#95A5A6")
        except Exception:
            return "#95A5A6"  # Gray fallback
    
    @classmethod
    def get_all_colors(cls) -> Dict[str, str]:
        """Get all step type colors for legend generation."""
        return cls.STEP_TYPE_COLORS.copy()
    
    @classmethod
    def get_color_legend(cls) -> str:
        """Get formatted color legend for display."""
        legend_lines = ["Step Type Color Legend:"]
        for step_type, color in cls.STEP_TYPE_COLORS.items():
            legend_lines.append(f"  {step_type}: {color}")
        return "\n".join(legend_lines)
