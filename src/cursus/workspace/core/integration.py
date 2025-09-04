"""
Workspace Integration Manager

Manages integration staging coordination and management for workspace components.
This module provides comprehensive integration staging, component promotion,
and cross-workspace integration validation.

Features:
- Integration staging coordination and management
- Component promotion and approval workflows
- Cross-workspace integration validation
- Integration readiness assessment
- Rollback and recovery capabilities
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class IntegrationStage(Enum):
    """Integration stage enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    INTEGRATION = "integration"
    PRODUCTION = "production"


class ComponentStatus(Enum):
    """Component status enumeration."""
    PENDING = "pending"
    STAGED = "staged"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


class StagedComponent:
    """Represents a staged component."""
    
    def __init__(
        self,
        component_id: str,
        source_workspace: str,
        component_type: str,
        stage: str = "staging",
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize staged component.
        
        Args:
            component_id: Component identifier
            source_workspace: Source workspace identifier
            component_type: Type of component (builder, script, etc.)
            stage: Current integration stage
            metadata: Additional component metadata
        """
        self.component_id = component_id
        self.source_workspace = source_workspace
        self.component_type = component_type
        self.stage = stage
        self.status = ComponentStatus.PENDING
        self.metadata = metadata or {}
        self.staged_at = datetime.now()
        self.last_updated = datetime.now()
        self.approval_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component_id": self.component_id,
            "source_workspace": self.source_workspace,
            "component_type": self.component_type,
            "stage": self.stage,
            "status": self.status.value,
            "metadata": self.metadata,
            "staged_at": self.staged_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "approval_history": self.approval_history,
            "validation_results": self.validation_results
        }


class IntegrationPipeline:
    """Represents an integration pipeline."""
    
    def __init__(self, pipeline_id: str, components: List[StagedComponent]):
        """
        Initialize integration pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            components: List of staged components
        """
        self.pipeline_id = pipeline_id
        self.components = components
        self.created_at = datetime.now()
        self.status = "pending"
        self.validation_results: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pipeline_id": self.pipeline_id,
            "components": [comp.to_dict() for comp in self.components],
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "validation_results": self.validation_results
        }


class WorkspaceIntegrationManager:
    """
    Integration staging coordination and management.
    
    Manages the integration staging process, component promotion,
    and cross-workspace integration validation.
    """
    
    def __init__(self, workspace_manager):
        """
        Initialize workspace integration manager.
        
        Args:
            workspace_manager: Parent WorkspaceManager instance
        """
        self.workspace_manager = workspace_manager
        
        # Staged components tracking
        self.staged_components: Dict[str, StagedComponent] = {}
        self.integration_pipelines: Dict[str, IntegrationPipeline] = {}
        
        # Integration staging directory
        self.staging_root = None
        if self.workspace_manager.workspace_root:
            self.staging_root = self.workspace_manager.workspace_root.parent / "development" / "review"
        
        logger.info("Initialized workspace integration manager")
    
    def stage_for_integration(
        self,
        component_id: str,
        source_workspace: str,
        target_stage: str = "integration"
    ) -> Dict[str, Any]:
        """
        Stage component for integration.
        
        Args:
            component_id: Component identifier
            source_workspace: Source workspace identifier
            target_stage: Target staging area
        
        Returns:
            Staging result information
        """
        logger.info(f"Staging component {component_id} from {source_workspace} to {target_stage}")
        
        staging_result = {
            "component_id": component_id,
            "source_workspace": source_workspace,
            "target_stage": target_stage,
            "success": False,
            "staged_at": None,
            "staging_path": None,
            "issues": []
        }
        
        try:
            # Validate component exists
            if not self._validate_component_exists(component_id, source_workspace):
                staging_result["issues"].append(f"Component not found: {component_id}")
                return staging_result
            
            # Determine component type
            component_type = self._determine_component_type(component_id, source_workspace)
            if not component_type:
                staging_result["issues"].append(f"Could not determine component type: {component_id}")
                return staging_result
            
            # Create staging directory if needed
            if not self.staging_root:
                staging_result["issues"].append("No staging root configured")
                return staging_result
            
            staging_path = self._create_staging_path(target_stage, component_id)
            staging_path.mkdir(parents=True, exist_ok=True)
            
            # Copy component to staging area
            copy_result = self._copy_component_to_staging(
                component_id, source_workspace, component_type, staging_path
            )
            
            if not copy_result["success"]:
                staging_result["issues"].extend(copy_result["issues"])
                return staging_result
            
            # Create staged component record
            staged_component = StagedComponent(
                component_id=component_id,
                source_workspace=source_workspace,
                component_type=component_type,
                stage=target_stage,
                metadata={
                    "staging_path": str(staging_path),
                    "original_files": copy_result["copied_files"]
                }
            )
            
            # Store staged component
            staging_key = f"{source_workspace}:{component_id}"
            self.staged_components[staging_key] = staged_component
            
            # Create staging metadata file
            self._create_staging_metadata(staged_component, staging_path)
            
            staging_result.update({
                "success": True,
                "staged_at": staged_component.staged_at.isoformat(),
                "staging_path": str(staging_path),
                "component_type": component_type
            })
            
            logger.info(f"Successfully staged component {component_id}")
            return staging_result
            
        except Exception as e:
            logger.error(f"Failed to stage component {component_id}: {e}")
            staging_result["issues"].append(f"Staging error: {e}")
            return staging_result
    
    def _validate_component_exists(self, component_id: str, source_workspace: str) -> bool:
        """Validate that component exists in source workspace."""
        try:
            # Use discovery manager to check component existence
            return self.workspace_manager.discovery_manager._check_component_exists(
                source_workspace, component_id
            )
        except Exception as e:
            logger.warning(f"Error validating component existence: {e}")
            return False
    
    def _determine_component_type(self, component_id: str, source_workspace: str) -> Optional[str]:
        """Determine the type of component."""
        try:
            # Get workspace path
            if source_workspace == "shared":
                workspace_path = self.workspace_manager.workspace_root / "shared"
            else:
                workspace_path = self.workspace_manager.workspace_root / "developers" / source_workspace
            
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            
            # Check each component type directory
            component_types = ["builders", "contracts", "specs", "scripts", "configs"]
            
            for comp_type in component_types:
                type_dir = cursus_dev_dir / comp_type
                if type_dir.exists():
                    # Look for component file
                    component_file = type_dir / f"{component_id}.py"
                    if component_file.exists():
                        return comp_type
            
            return None
            
        except Exception as e:
            logger.warning(f"Error determining component type: {e}")
            return None
    
    def _create_staging_path(self, target_stage: str, component_id: str) -> Path:
        """Create staging path for component."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_dir = f"{component_id}_{timestamp}"
        return self.staging_root / target_stage / staging_dir
    
    def _copy_component_to_staging(
        self,
        component_id: str,
        source_workspace: str,
        component_type: str,
        staging_path: Path
    ) -> Dict[str, Any]:
        """Copy component files to staging area."""
        copy_result = {
            "success": False,
            "copied_files": [],
            "issues": []
        }
        
        try:
            # Get source workspace path
            if source_workspace == "shared":
                workspace_path = self.workspace_manager.workspace_root / "shared"
            else:
                workspace_path = self.workspace_manager.workspace_root / "developers" / source_workspace
            
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            source_dir = cursus_dev_dir / component_type
            
            # Find component files
            component_files = []
            
            # Main component file
            main_file = source_dir / f"{component_id}.py"
            if main_file.exists():
                component_files.append(main_file)
            
            # Look for related files (e.g., test files, config files)
            related_patterns = [
                f"{component_id}_*.py",
                f"*_{component_id}.py",
                f"{component_id}.json",
                f"{component_id}.yaml"
            ]
            
            for pattern in related_patterns:
                component_files.extend(source_dir.glob(pattern))
            
            if not component_files:
                copy_result["issues"].append(f"No files found for component: {component_id}")
                return copy_result
            
            # Copy files to staging area
            for source_file in component_files:
                try:
                    dest_file = staging_path / source_file.name
                    shutil.copy2(source_file, dest_file)
                    copy_result["copied_files"].append({
                        "source": str(source_file),
                        "destination": str(dest_file),
                        "size": source_file.stat().st_size
                    })
                except Exception as e:
                    copy_result["issues"].append(f"Failed to copy {source_file}: {e}")
            
            if copy_result["copied_files"]:
                copy_result["success"] = True
            
            return copy_result
            
        except Exception as e:
            copy_result["issues"].append(f"Copy operation failed: {e}")
            return copy_result
    
    def _create_staging_metadata(self, staged_component: StagedComponent, staging_path: Path) -> None:
        """Create metadata file for staged component."""
        try:
            metadata_file = staging_path / "staging_metadata.json"
            metadata = {
                "component_info": staged_component.to_dict(),
                "staging_info": {
                    "staging_path": str(staging_path),
                    "created_by": "WorkspaceIntegrationManager",
                    "created_at": datetime.now().isoformat()
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to create staging metadata: {e}")
    
    def validate_integration_readiness(
        self,
        staged_components: List[str]
    ) -> Dict[str, Any]:
        """
        Validate integration readiness for staged components.
        
        Args:
            staged_components: List of staged component identifiers
        
        Returns:
            Integration readiness validation results
        """
        logger.info(f"Validating integration readiness for {len(staged_components)} components")
        
        validation_result = {
            "components": staged_components,
            "overall_ready": True,
            "component_results": {},
            "integration_issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Validate each staged component
            for component_key in staged_components:
                if component_key not in self.staged_components:
                    validation_result["overall_ready"] = False
                    validation_result["integration_issues"].append(f"Component not found in staging: {component_key}")
                    continue
                
                staged_component = self.staged_components[component_key]
                component_validation = self._validate_staged_component(staged_component)
                validation_result["component_results"][component_key] = component_validation
                
                if not component_validation["ready"]:
                    validation_result["overall_ready"] = False
                    validation_result["integration_issues"].extend(component_validation["issues"])
                
                validation_result["warnings"].extend(component_validation.get("warnings", []))
            
            # Cross-component validation
            if validation_result["overall_ready"]:
                cross_validation = self._validate_cross_component_compatibility(staged_components)
                validation_result["cross_component_validation"] = cross_validation
                
                if not cross_validation["compatible"]:
                    validation_result["overall_ready"] = False
                    validation_result["integration_issues"].extend(cross_validation["issues"])
            
            # Generate recommendations
            if not validation_result["overall_ready"]:
                validation_result["recommendations"] = self._generate_integration_recommendations(
                    validation_result
                )
            
            logger.info(f"Integration readiness validation completed: {'READY' if validation_result['overall_ready'] else 'NOT READY'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate integration readiness: {e}")
            validation_result["overall_ready"] = False
            validation_result["integration_issues"].append(f"Validation error: {e}")
            return validation_result
    
    def _validate_staged_component(self, staged_component: StagedComponent) -> Dict[str, Any]:
        """Validate individual staged component."""
        validation = {
            "component_id": staged_component.component_id,
            "ready": True,
            "issues": [],
            "warnings": [],
            "checks_performed": []
        }
        
        try:
            # Check 1: Staging path exists
            staging_path = Path(staged_component.metadata.get("staging_path", ""))
            if not staging_path.exists():
                validation["ready"] = False
                validation["issues"].append("Staging path does not exist")
            else:
                validation["checks_performed"].append("staging_path_exists")
            
            # Check 2: Component files exist
            if staging_path.exists():
                copied_files = staged_component.metadata.get("original_files", [])
                for file_info in copied_files:
                    dest_path = Path(file_info["destination"])
                    if not dest_path.exists():
                        validation["ready"] = False
                        validation["issues"].append(f"Staged file missing: {dest_path}")
                
                validation["checks_performed"].append("staged_files_exist")
            
            # Check 3: Component syntax validation
            if staging_path.exists():
                syntax_check = self._validate_component_syntax(staging_path)
                if not syntax_check["valid"]:
                    validation["ready"] = False
                    validation["issues"].extend(syntax_check["issues"])
                validation["warnings"].extend(syntax_check.get("warnings", []))
                validation["checks_performed"].append("syntax_validation")
            
            # Check 4: Component age (warn if too old)
            component_age = datetime.now() - staged_component.staged_at
            if component_age > timedelta(days=7):
                validation["warnings"].append(f"Component staged {component_age.days} days ago")
            
            return validation
            
        except Exception as e:
            validation["ready"] = False
            validation["issues"].append(f"Component validation error: {e}")
            return validation
    
    def _validate_component_syntax(self, staging_path: Path) -> Dict[str, Any]:
        """Validate syntax of staged component files."""
        syntax_validation = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            # Check Python files for syntax errors
            python_files = list(staging_path.glob("*.py"))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Basic syntax check using compile
                    compile(content, str(py_file), 'exec')
                    
                except SyntaxError as e:
                    syntax_validation["valid"] = False
                    syntax_validation["issues"].append(f"Syntax error in {py_file}: {e}")
                except Exception as e:
                    syntax_validation["warnings"].append(f"Could not validate {py_file}: {e}")
            
            return syntax_validation
            
        except Exception as e:
            syntax_validation["valid"] = False
            syntax_validation["issues"].append(f"Syntax validation error: {e}")
            return syntax_validation
    
    def _validate_cross_component_compatibility(self, staged_components: List[str]) -> Dict[str, Any]:
        """Validate compatibility between staged components."""
        compatibility = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "compatibility_matrix": {}
        }
        
        try:
            # Check for naming conflicts
            component_names = set()
            for component_key in staged_components:
                if component_key in self.staged_components:
                    staged_comp = self.staged_components[component_key]
                    comp_name = staged_comp.component_id
                    
                    if comp_name in component_names:
                        compatibility["compatible"] = False
                        compatibility["issues"].append(f"Component name conflict: {comp_name}")
                    else:
                        component_names.add(comp_name)
            
            # Check for dependency conflicts
            # This would require more sophisticated dependency analysis
            # For now, we'll do basic checks
            
            return compatibility
            
        except Exception as e:
            compatibility["compatible"] = False
            compatibility["issues"].append(f"Cross-component validation error: {e}")
            return compatibility
    
    def _generate_integration_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for integration issues."""
        recommendations = []
        
        try:
            issues = validation_result.get("integration_issues", [])
            
            for issue in issues:
                if "not found in staging" in issue:
                    recommendations.append("Re-stage missing components before integration")
                elif "Staging path does not exist" in issue:
                    recommendations.append("Verify staging area integrity and re-stage if necessary")
                elif "Syntax error" in issue:
                    recommendations.append("Fix syntax errors in component code before integration")
                elif "name conflict" in issue:
                    recommendations.append("Resolve component naming conflicts")
            
            # General recommendations
            if len(issues) > 5:
                recommendations.append("Consider staging components in smaller batches")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["Review integration issues and consult documentation"]
    
    def promote_to_production(self, component_id: str) -> Dict[str, Any]:
        """
        Promote component to production.
        
        Args:
            component_id: Component identifier to promote
        
        Returns:
            Promotion result information
        """
        logger.info(f"Promoting component to production: {component_id}")
        
        promotion_result = {
            "component_id": component_id,
            "success": False,
            "promoted_at": None,
            "production_path": None,
            "issues": []
        }
        
        try:
            # Find staged component
            staged_component = None
            for key, comp in self.staged_components.items():
                if comp.component_id == component_id:
                    staged_component = comp
                    break
            
            if not staged_component:
                promotion_result["issues"].append(f"Component not found in staging: {component_id}")
                return promotion_result
            
            # Validate component is ready for promotion
            validation = self._validate_staged_component(staged_component)
            if not validation["ready"]:
                promotion_result["issues"].append("Component not ready for promotion")
                promotion_result["issues"].extend(validation["issues"])
                return promotion_result
            
            # Copy to production area (shared workspace)
            production_result = self._copy_to_production(staged_component)
            if not production_result["success"]:
                promotion_result["issues"].extend(production_result["issues"])
                return promotion_result
            
            # Update component status
            staged_component.status = ComponentStatus.PROMOTED
            staged_component.last_updated = datetime.now()
            staged_component.approval_history.append({
                "action": "promoted",
                "timestamp": datetime.now().isoformat(),
                "details": "Promoted to production"
            })
            
            promotion_result.update({
                "success": True,
                "promoted_at": staged_component.last_updated.isoformat(),
                "production_path": production_result["production_path"]
            })
            
            logger.info(f"Successfully promoted component {component_id}")
            return promotion_result
            
        except Exception as e:
            logger.error(f"Failed to promote component {component_id}: {e}")
            promotion_result["issues"].append(f"Promotion error: {e}")
            return promotion_result
    
    def _copy_to_production(self, staged_component: StagedComponent) -> Dict[str, Any]:
        """Copy staged component to production (shared workspace)."""
        copy_result = {
            "success": False,
            "production_path": None,
            "issues": []
        }
        
        try:
            # Get production path (shared workspace)
            shared_workspace = self.workspace_manager.workspace_root / "shared"
            if not shared_workspace.exists():
                copy_result["issues"].append("Shared workspace does not exist")
                return copy_result
            
            production_dir = shared_workspace / "src" / "cursus_dev" / "steps" / staged_component.component_type
            production_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files from staging to production
            staging_path = Path(staged_component.metadata["staging_path"])
            copied_files = staged_component.metadata.get("original_files", [])
            
            for file_info in copied_files:
                source_file = Path(file_info["destination"])  # File in staging
                if source_file.exists():
                    dest_file = production_dir / source_file.name
                    shutil.copy2(source_file, dest_file)
            
            copy_result.update({
                "success": True,
                "production_path": str(production_dir)
            })
            
            return copy_result
            
        except Exception as e:
            copy_result["issues"].append(f"Production copy failed: {e}")
            return copy_result
    
    def rollback_integration(self, component_id: str) -> Dict[str, Any]:
        """
        Rollback component integration.
        
        Args:
            component_id: Component identifier to rollback
        
        Returns:
            Rollback result information
        """
        logger.info(f"Rolling back component integration: {component_id}")
        
        rollback_result = {
            "component_id": component_id,
            "success": False,
            "rolled_back_at": None,
            "issues": []
        }
        
        try:
            # Find staged component
            staged_component = None
            for key, comp in self.staged_components.items():
                if comp.component_id == component_id:
                    staged_component = comp
                    break
            
            if not staged_component:
                rollback_result["issues"].append(f"Component not found: {component_id}")
                return rollback_result
            
            # Remove from production if promoted
            if staged_component.status == ComponentStatus.PROMOTED:
                self._remove_from_production(staged_component)
            
            # Update component status
            staged_component.status = ComponentStatus.ROLLED_BACK
            staged_component.last_updated = datetime.now()
            staged_component.approval_history.append({
                "action": "rolled_back",
                "timestamp": datetime.now().isoformat(),
                "details": "Integration rolled back"
            })
            
            rollback_result.update({
                "success": True,
                "rolled_back_at": staged_component.last_updated.isoformat()
            })
            
            logger.info(f"Successfully rolled back component {component_id}")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Failed to rollback component {component_id}: {e}")
            rollback_result["issues"].append(f"Rollback error: {e}")
            return rollback_result
    
    def _remove_from_production(self, staged_component: StagedComponent) -> None:
        """Remove component from production."""
        try:
            shared_workspace = self.workspace_manager.workspace_root / "shared"
            production_dir = shared_workspace / "src" / "cursus_dev" / "steps" / staged_component.component_type
            
            # Remove component files
            component_file = production_dir / f"{staged_component.component_id}.py"
            if component_file.exists():
                component_file.unlink()
            
            # Remove related files
            related_files = list(production_dir.glob(f"{staged_component.component_id}_*.py"))
            related_files.extend(production_dir.glob(f"*_{staged_component.component_id}.py"))
            
            for file_path in related_files:
                file_path.unlink()
                
        except Exception as e:
            logger.warning(f"Error removing component from production: {e}")
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of integration activities."""
        try:
            return {
                "staged_components": len(self.staged_components),
                "integration_pipelines": len(self.integration_pipelines),
                "component_status_distribution": {
                    status.value: len([
                        comp for comp in self.staged_components.values()
                        if comp.status == status
                    ])
                    for status in ComponentStatus
                },
                "recent_activities": self._get_recent_activities()
            }
        except Exception as e:
            logger.error(f"Failed to get integration summary: {e}")
            return {"error": str(e)}
    
    def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """Get recent integration activities."""
        try:
            activities = []
            
            # Get recent component activities
            for comp in self.staged_components.values():
                for activity in comp.approval_history[-3:]:  # Last 3 activities
                    activities.append({
                        "component_id": comp.component_id,
                        "activity": activity,
                        "timestamp": activity["timestamp"]
                    })
            
            # Sort by timestamp (most recent first)
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return activities[:10]  # Return last 10 activities
            
        except Exception as e:
            logger.warning(f"Error getting recent activities: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get integration management statistics."""
        try:
            return {
                "integration_operations": {
                    "total_staged_components": len(self.staged_components),
                    "active_pipelines": len(self.integration_pipelines),
                    "promotion_success_rate": self._calculate_promotion_success_rate()
                },
                "component_statistics": {
                    "by_type": self._get_component_type_statistics(),
                    "by_status": self._get_component_status_statistics(),
                    "by_workspace": self._get_workspace_statistics()
                },
                "integration_summary": self.get_integration_summary()
            }
        except Exception as e:
            logger.error(f"Failed to get integration statistics: {e}")
            return {"error": str(e)}
    
    def _calculate_promotion_success_rate(self) -> float:
        """Calculate promotion success rate."""
        try:
            total_promotions = len([
                comp for comp in self.staged_components.values()
                if comp.status in [ComponentStatus.PROMOTED, ComponentStatus.ROLLED_BACK]
            ])
            
            successful_promotions = len([
                comp for comp in self.staged_components.values()
                if comp.status == ComponentStatus.PROMOTED
            ])
            
            if total_promotions == 0:
                return 0.0
            
            return successful_promotions / total_promotions
            
        except Exception:
            return 0.0
    
    def _get_component_type_statistics(self) -> Dict[str, int]:
        """Get statistics by component type."""
        try:
            type_stats = {}
            for comp in self.staged_components.values():
                comp_type = comp.component_type
                type_stats[comp_type] = type_stats.get(comp_type, 0) + 1
            return type_stats
        except Exception:
            return {}
    
    def _get_component_status_statistics(self) -> Dict[str, int]:
        """Get statistics by component status."""
        try:
            status_stats = {}
            for comp in self.staged_components.values():
                status = comp.status.value
                status_stats[status] = status_stats.get(status, 0) + 1
            return status_stats
        except Exception:
            return {}
    
    def _get_workspace_statistics(self) -> Dict[str, int]:
        """Get statistics by source workspace."""
        try:
            workspace_stats = {}
            for comp in self.staged_components.values():
                workspace = comp.source_workspace
                workspace_stats[workspace] = workspace_stats.get(workspace, 0) + 1
            return workspace_stats
        except Exception:
            return {}
