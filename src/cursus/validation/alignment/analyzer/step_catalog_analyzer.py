"""
Unified Step Catalog Analyzer

Provides comprehensive step analysis using StepCatalog integration to eliminate
redundancy and leverage built-in discovery capabilities.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .config_analyzer import ConfigurationAnalyzer
from .builder_analyzer import BuilderCodeAnalyzer
from .import_analyzer import ImportAnalyzer


class StepCatalogAnalyzer:
    """
    Unified analyzer that leverages StepCatalog for comprehensive step analysis.
    
    This analyzer consolidates the functionality of individual analyzers while
    using StepCatalog's built-in capabilities for:
    - Direct class loading
    - Framework detection
    - Metadata access
    - Workspace-aware discovery
    """
    
    def __init__(self, step_catalog):
        """
        Initialize the unified analyzer with StepCatalog integration.
        
        Args:
            step_catalog: StepCatalog instance for enhanced analysis
        """
        self.step_catalog = step_catalog
        
        # Initialize component analyzers with StepCatalog integration
        self.config_analyzer = ConfigurationAnalyzer()  # Already optimized
        self.builder_analyzer = BuilderCodeAnalyzer(step_catalog=step_catalog)
        self.import_analyzer = ImportAnalyzer(step_catalog=step_catalog)
    
    def analyze_step(self, step_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a step using StepCatalog integration.
        
        Args:
            step_name: Name of the step to analyze
            
        Returns:
            Dictionary containing comprehensive step analysis
        """
        try:
            # Get step information from StepCatalog
            step_info = self.step_catalog.get_step_info(step_name)
            if not step_info:
                return {"error": f"Step {step_name} not found in StepCatalog"}
            
            # Base analysis with StepCatalog metadata
            analysis = {
                "step_name": step_info.step_name,
                "workspace_id": step_info.workspace_id,
                "registry_data": step_info.registry_data,
                "file_components": {
                    comp_type: {
                        "path": str(metadata.path),
                        "file_type": metadata.file_type,
                        "last_modified": metadata.modified_time.isoformat() if metadata.modified_time else None
                    }
                    for comp_type, metadata in step_info.file_components.items()
                    if metadata
                },
                
                # StepCatalog built-in capabilities
                "framework": self.step_catalog.detect_framework(step_name),
                "available_components": list(step_info.file_components.keys()),
            }
            
            # Component-specific analysis
            analysis.update({
                "config_analysis": self._analyze_config_component(step_name),
                "builder_analysis": self._analyze_builder_component(step_name),
                "import_analysis": self._analyze_import_component(step_name),
                "alignment_summary": self._generate_alignment_summary(step_name, step_info),
            })
            
            return analysis
            
        except Exception as e:
            return {
                "error": str(e),
                "step_name": step_name,
                "config_analysis": {},
                "builder_analysis": {},
                "import_analysis": {},
            }
    
    def analyze_multiple_steps(self, step_names: list) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple steps efficiently using StepCatalog bulk operations.
        
        Args:
            step_names: List of step names to analyze
            
        Returns:
            Dictionary mapping step names to their analysis results
        """
        results = {}
        
        for step_name in step_names:
            try:
                results[step_name] = self.analyze_step(step_name)
            except Exception as e:
                results[step_name] = {
                    "error": str(e),
                    "step_name": step_name
                }
        
        return results
    
    def _analyze_config_component(self, step_name: str) -> Dict[str, Any]:
        """
        Analyze configuration component using StepCatalog integration.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Configuration analysis results
        """
        try:
            # Use StepCatalog's config discovery capabilities
            config_classes = self.step_catalog.discover_config_classes()
            
            # Try different naming patterns for config class
            config_class = None
            possible_names = [
                f"{step_name}Config",
                f"{step_name.title()}Config", 
                step_name,
                f"Config{step_name}",
                f"Config{step_name.title()}"
            ]
            
            for name in possible_names:
                if name in config_classes:
                    config_class = config_classes[name]
                    break
            
            if not config_class:
                return {"error": f"No configuration class found for step {step_name}. Tried: {possible_names}"}
            
            # Analyze using ConfigurationAnalyzer (already optimized)
            return self.config_analyzer.analyze_config_class(config_class, config_class.__name__)
            
        except Exception as e:
            return {"error": f"Config analysis failed: {str(e)}"}
    
    def _analyze_builder_component(self, step_name: str) -> Dict[str, Any]:
        """
        Analyze builder component using StepCatalog integration.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Builder analysis results
        """
        try:
            # Use BuilderCodeAnalyzer's StepCatalog-integrated method
            return self.builder_analyzer.analyze_builder_step(step_name)
            
        except Exception as e:
            return {"error": f"Builder analysis failed: {str(e)}"}
    
    def _analyze_import_component(self, step_name: str) -> Dict[str, Any]:
        """
        Analyze import component using StepCatalog integration.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Import analysis results
        """
        try:
            # Use ImportAnalyzer's StepCatalog-integrated method
            return self.import_analyzer.analyze_step_imports(step_name)
            
        except Exception as e:
            return {"error": f"Import analysis failed: {str(e)}"}
    
    def _generate_alignment_summary(self, step_name: str, step_info) -> Dict[str, Any]:
        """
        Generate alignment summary using StepCatalog metadata.
        
        Args:
            step_name: Name of the step
            step_info: StepInfo object from StepCatalog
            
        Returns:
            Alignment summary
        """
        try:
            # Check component availability
            available_components = list(step_info.file_components.keys())
            
            # Standard component expectations
            expected_components = ['script', 'contract', 'spec', 'builder', 'config']
            
            alignment_summary = {
                "component_coverage": {
                    "available": available_components,
                    "missing": [comp for comp in expected_components if comp not in available_components],
                    "coverage_percentage": (len(available_components) / len(expected_components)) * 100
                },
                "framework_alignment": {
                    "detected_framework": self.step_catalog.detect_framework(step_name),
                    "registry_framework": step_info.registry_data.get('framework'),
                    "consistent": (
                        self.step_catalog.detect_framework(step_name) == 
                        step_info.registry_data.get('framework')
                    ) if step_info.registry_data.get('framework') else None
                },
                "workspace_context": {
                    "workspace_id": step_info.workspace_id,
                    "is_workspace_component": step_info.workspace_id != "core",
                    "registry_data_available": bool(step_info.registry_data)
                }
            }
            
            return alignment_summary
            
        except Exception as e:
            return {"error": f"Alignment summary generation failed: {str(e)}"}
    
    def get_analyzer_summary(self) -> Dict[str, Any]:
        """
        Get summary of analyzer capabilities and StepCatalog integration status.
        
        Returns:
            Summary of analyzer status and capabilities
        """
        return {
            "step_catalog_available": self.step_catalog is not None,
            "step_catalog_metrics": self.step_catalog.get_metrics_report() if self.step_catalog else None,
            "analyzer_components": {
                "config_analyzer": "ConfigurationAnalyzer (StepCatalog-optimized)",
                "builder_analyzer": "BuilderCodeAnalyzer (StepCatalog-integrated)",
                "import_analyzer": "ImportAnalyzer (StepCatalog-integrated)"
            },
            "capabilities": [
                "Direct class loading via StepCatalog",
                "Built-in framework detection",
                "Metadata-driven analysis",
                "Workspace-aware discovery",
                "Registry integration",
                "Bulk analysis operations"
            ]
        }
    
    def analyze_workspace_steps(self, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze all steps in a workspace using StepCatalog's workspace awareness.
        
        Args:
            workspace_id: Optional workspace ID filter
            
        Returns:
            Analysis results for all steps in the workspace
        """
        try:
            # Get steps from workspace using StepCatalog
            available_steps = self.step_catalog.list_available_steps(workspace_id=workspace_id)
            
            if not available_steps:
                return {
                    "error": f"No steps found in workspace {workspace_id}",
                    "workspace_id": workspace_id
                }
            
            # Analyze all steps in the workspace
            workspace_analysis = {
                "workspace_id": workspace_id or "all",
                "total_steps": len(available_steps),
                "step_analyses": self.analyze_multiple_steps(available_steps),
                "workspace_summary": self._generate_workspace_summary(available_steps)
            }
            
            return workspace_analysis
            
        except Exception as e:
            return {
                "error": f"Workspace analysis failed: {str(e)}",
                "workspace_id": workspace_id
            }
    
    def _generate_workspace_summary(self, step_names: list) -> Dict[str, Any]:
        """
        Generate summary statistics for a workspace.
        
        Args:
            step_names: List of step names in the workspace
            
        Returns:
            Workspace summary statistics
        """
        try:
            summary = {
                "total_steps": len(step_names),
                "framework_distribution": {},
                "component_coverage": {
                    "script": 0, "contract": 0, "spec": 0, "builder": 0, "config": 0
                },
                "workspace_distribution": {}
            }
            
            for step_name in step_names:
                step_info = self.step_catalog.get_step_info(step_name)
                if step_info:
                    # Framework distribution
                    framework = self.step_catalog.detect_framework(step_name)
                    if framework:
                        summary["framework_distribution"][framework] = (
                            summary["framework_distribution"].get(framework, 0) + 1
                        )
                    
                    # Component coverage
                    for component_type in summary["component_coverage"]:
                        if component_type in step_info.file_components:
                            summary["component_coverage"][component_type] += 1
                    
                    # Workspace distribution
                    workspace_id = step_info.workspace_id
                    summary["workspace_distribution"][workspace_id] = (
                        summary["workspace_distribution"].get(workspace_id, 0) + 1
                    )
            
            return summary
            
        except Exception as e:
            return {"error": f"Workspace summary generation failed: {str(e)}"}
