"""Data flow manager for tracking and managing data between script executions."""

import json
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..utils.error_handling import DataFlowError

class DataFlowManager:
    """Basic data flow manager for script executions (Phase 1 implementation)"""
    
    def __init__(self, workspace_dir: str):
        """Initialize data flow manager with workspace directory"""
        self.workspace_dir = Path(workspace_dir)
        self.data_lineage = []
        
        # Create directories if they don't exist
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "inputs").mkdir(exist_ok=True)
        (self.workspace_dir / "outputs").mkdir(exist_ok=True)
        (self.workspace_dir / "metadata").mkdir(exist_ok=True)
    
    def setup_step_inputs(self, step_name: str, upstream_outputs: Dict[str, str], 
                         input_spec: Optional[Dict] = None) -> Dict[str, str]:
        """Map upstream outputs to current step inputs based on specifications"""
        
        if not step_name:
            raise DataFlowError("Step name cannot be empty")
        
        # Phase 1: Basic implementation - direct mapping
        if not upstream_outputs:
            # No upstream outputs, create empty input directory
            input_dir = self.workspace_dir / "inputs" / step_name
            try:
                input_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise DataFlowError(f"Failed to create input directory for step '{step_name}': {e}")
            return {"input": str(input_dir)}
        
        # Validate that upstream output paths exist
        for output_name, output_path in upstream_outputs.items():
            if not os.path.exists(output_path):
                raise DataFlowError(f"Upstream output '{output_name}' does not exist at path: {output_path}")
            
        # Direct pass-through in Phase 1
        return upstream_outputs
    
    def capture_step_outputs(self, step_name: str, output_paths: Dict[str, str]) -> Dict[str, Any]:
        """Capture and validate step outputs after execution"""
        
        if not step_name:
            raise DataFlowError("Step name cannot be empty")
        
        if not output_paths:
            raise DataFlowError(f"No output paths provided for step '{step_name}'")
        
        # Validate that all output paths exist
        for output_name, output_path in output_paths.items():
            if not os.path.exists(output_path):
                raise DataFlowError(f"Output '{output_name}' does not exist at path: {output_path}")
        
        # Phase 1: Basic implementation - just return output paths
        metadata = {
            "step_name": step_name,
            "output_paths": output_paths,
            "captured_at": str(datetime.now()),
        }
        
        # Save metadata
        metadata_path = self.workspace_dir / "metadata" / f"{step_name}_outputs.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except (OSError, IOError) as e:
            raise DataFlowError(f"Failed to save metadata for step '{step_name}': {e}")
            
        # Record in data lineage
        self.data_lineage.append({
            "step": step_name,
            "outputs": output_paths,
            "metadata_file": str(metadata_path)
        })
        
        return output_paths
    
    def track_data_lineage(self, step_name: str, inputs: Dict[str, str], 
                         outputs: Dict[str, str]) -> None:
        """Track data lineage for a step execution"""
        
        if not step_name:
            raise DataFlowError("Step name cannot be empty for lineage tracking")
        
        lineage_entry = {
            "step_name": step_name,
            "inputs": inputs,
            "outputs": outputs,
            "timestamp": str(datetime.now())
        }
        
        self.data_lineage.append(lineage_entry)
        
        # Save to file
        lineage_file = self.workspace_dir / "metadata" / "data_lineage.json"
        
        if lineage_file.exists():
            try:
                with open(lineage_file, "r") as f:
                    lineage = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                raise DataFlowError(f"Failed to read existing lineage file: {e}")
        else:
            lineage = []
            
        lineage.append(lineage_entry)
        
        try:
            with open(lineage_file, "w") as f:
                json.dump(lineage, f, indent=2)
        except (OSError, IOError) as e:
            raise DataFlowError(f"Failed to save lineage data: {e}")
