"""
Workspace component inventory management.

This module provides data structures for tracking and organizing components
discovered across multiple developer workspaces, maintaining separation of
concerns from the unified step catalog system.
"""

import logging
from typing import Dict, Any, Set, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class ComponentInventory:
    """
    Inventory of discovered workspace components.
    
    Maintains workspace-specific business logic for component organization
    and tracking, separate from the unified step catalog discovery system.
    
    DESIGN PRINCIPLES:
    - Single Responsibility: Focuses on component inventory management
    - Separation of Concerns: Workspace business logic separate from discovery
    - Explicit Dependencies: Clear interface for component tracking
    """

    def __init__(self):
        """Initialize component inventory."""
        self.builders: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.contracts: Dict[str, Dict[str, Any]] = {}
        self.specs: Dict[str, Dict[str, Any]] = {}
        self.scripts: Dict[str, Dict[str, Any]] = {}
        self.summary: Dict[str, Any] = {
            "total_components": 0,
            "developers": [],
            "step_types": set(),
            "component_types": {}
        }

    def add_component(
        self, component_type: str, component_id: str, component_info: Dict[str, Any]
    ) -> None:
        """
        Add component to inventory.
        
        Args:
            component_type: Type of component (builders, configs, contracts, specs, scripts)
            component_id: Unique identifier for the component
            component_info: Component metadata and information
        """
        if component_type == "builders":
            self.builders[component_id] = component_info
        elif component_type == "configs":
            self.configs[component_id] = component_info
        elif component_type == "contracts":
            self.contracts[component_id] = component_info
        elif component_type == "specs":
            self.specs[component_id] = component_info
        elif component_type == "scripts":
            self.scripts[component_id] = component_info
        else:
            logger.warning(f"Unknown component type: {component_type}")
            return

        # Update summary statistics
        self.summary["total_components"] += 1
        
        # Track developers
        developer_id = component_info.get("developer_id")
        if developer_id and developer_id not in self.summary["developers"]:
            self.summary["developers"].append(developer_id)
        
        # Track step types
        step_type = component_info.get("step_type")
        if step_type:
            self.summary["step_types"].add(step_type)
        
        # Update component type counts
        if component_type not in self.summary["component_types"]:
            self.summary["component_types"][component_type] = 0
        self.summary["component_types"][component_type] += 1

    def get_components_by_type(self, component_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all components of a specific type."""
        if component_type == "builders":
            return self.builders.copy()
        elif component_type == "configs":
            return self.configs.copy()
        elif component_type == "contracts":
            return self.contracts.copy()
        elif component_type == "specs":
            return self.specs.copy()
        elif component_type == "scripts":
            return self.scripts.copy()
        else:
            return {}

    def get_components_by_developer(self, developer_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all components for a specific developer."""
        developer_components = {
            "builders": [],
            "configs": [],
            "contracts": [],
            "specs": [],
            "scripts": []
        }
        
        for component_type in developer_components.keys():
            components = self.get_components_by_type(component_type)
            for component_id, component_info in components.items():
                if component_info.get("developer_id") == developer_id:
                    developer_components[component_type].append({
                        "component_id": component_id,
                        **component_info
                    })
        
        return developer_components

    def get_components_by_step_type(self, step_type: str) -> List[Dict[str, Any]]:
        """Get all components for a specific step type."""
        matching_components = []
        
        for component_type in ["builders", "configs", "contracts", "specs", "scripts"]:
            components = self.get_components_by_type(component_type)
            for component_id, component_info in components.items():
                if component_info.get("step_type") == step_type:
                    matching_components.append({
                        "component_id": component_id,
                        "component_type": component_type,
                        **component_info
                    })
        
        return matching_components

    def remove_component(self, component_type: str, component_id: str) -> bool:
        """Remove a component from the inventory."""
        components = self.get_components_by_type(component_type)
        if component_id in components:
            if component_type == "builders":
                del self.builders[component_id]
            elif component_type == "configs":
                del self.configs[component_id]
            elif component_type == "contracts":
                del self.contracts[component_id]
            elif component_type == "specs":
                del self.specs[component_id]
            elif component_type == "scripts":
                del self.scripts[component_id]
            
            # Update summary
            self.summary["total_components"] -= 1
            if component_type in self.summary["component_types"]:
                self.summary["component_types"][component_type] -= 1
            
            return True
        return False

    def clear(self) -> None:
        """Clear all components from inventory."""
        self.builders.clear()
        self.configs.clear()
        self.contracts.clear()
        self.specs.clear()
        self.scripts.clear()
        self.summary = {
            "total_components": 0,
            "developers": [],
            "step_types": set(),
            "component_types": {}
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert inventory to dictionary for serialization.
        
        Returns:
            Dictionary representation of the inventory
        """
        return {
            "builders": self.builders,
            "configs": self.configs,
            "contracts": self.contracts,
            "specs": self.specs,
            "scripts": self.scripts,
            "summary": {
                **self.summary,
                "step_types": list(self.summary["step_types"])  # Convert set to list
            },
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load inventory from dictionary.
        
        Args:
            data: Dictionary representation to load from
        """
        self.builders = data.get("builders", {})
        self.configs = data.get("configs", {})
        self.contracts = data.get("contracts", {})
        self.specs = data.get("specs", {})
        self.scripts = data.get("scripts", {})
        
        summary = data.get("summary", {})
        self.summary = {
            "total_components": summary.get("total_components", 0),
            "developers": summary.get("developers", []),
            "step_types": set(summary.get("step_types", [])),  # Convert list to set
            "component_types": summary.get("component_types", {})
        }

    def merge(self, other: 'ComponentInventory') -> None:
        """
        Merge another inventory into this one.
        
        Args:
            other: Another ComponentInventory to merge
        """
        # Merge each component type
        self.builders.update(other.builders)
        self.configs.update(other.configs)
        self.contracts.update(other.contracts)
        self.specs.update(other.specs)
        self.scripts.update(other.scripts)
        
        # Recalculate summary
        self._recalculate_summary()

    def _recalculate_summary(self) -> None:
        """Recalculate summary statistics from current components."""
        all_developers = set()
        all_step_types = set()
        component_type_counts = defaultdict(int)
        
        for component_type in ["builders", "configs", "contracts", "specs", "scripts"]:
            components = self.get_components_by_type(component_type)
            component_type_counts[component_type] = len(components)
            
            for component_info in components.values():
                if component_info.get("developer_id"):
                    all_developers.add(component_info["developer_id"])
                if component_info.get("step_type"):
                    all_step_types.add(component_info["step_type"])
        
        self.summary = {
            "total_components": sum(component_type_counts.values()),
            "developers": list(all_developers),
            "step_types": all_step_types,
            "component_types": dict(component_type_counts)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the inventory."""
        stats = {
            "total_components": self.summary["total_components"],
            "total_developers": len(self.summary["developers"]),
            "total_step_types": len(self.summary["step_types"]),
            "component_distribution": self.summary["component_types"].copy(),
            "developers": self.summary["developers"].copy(),
            "step_types": list(self.summary["step_types"]),
        }
        
        # Calculate averages
        if stats["total_developers"] > 0:
            stats["avg_components_per_developer"] = stats["total_components"] / stats["total_developers"]
        else:
            stats["avg_components_per_developer"] = 0
        
        return stats

    def __len__(self) -> int:
        """Return total number of components."""
        return self.summary["total_components"]

    def __repr__(self) -> str:
        """String representation of the inventory."""
        return (f"ComponentInventory(total={self.summary['total_components']}, "
                f"developers={len(self.summary['developers'])}, "
                f"step_types={len(self.summary['step_types'])})")
