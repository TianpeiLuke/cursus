"""
Core Alignment Testers Module

This module contains the core alignment validation logic for all four alignment levels.
Each tester focuses on a specific alignment relationship in the pipeline architecture.

Alignment Levels:
1. Script ↔ Contract Alignment (Level 1)
2. Contract ↔ Specification Alignment (Level 2)  
3. Specification ↔ Dependencies Alignment (Level 3)
4. Builder ↔ Configuration Alignment (Level 4)

Components:
- script_contract_alignment.py: Level 1 - Script and contract alignment validation
- contract_spec_alignment.py: Level 2 - Contract and specification alignment validation
- spec_dependency_alignment.py: Level 3 - Specification and dependency alignment validation
- builder_config_alignment.py: Level 4 - Builder and configuration alignment validation
"""

# Level 1: Script ↔ Contract Alignment
from .script_contract_alignment import ScriptContractAlignmentTester

# Level 2: Contract ↔ Specification Alignment
from .contract_spec_alignment import ContractSpecificationAlignmentTester

# Level 3: Specification ↔ Dependencies Alignment
from .spec_dependency_alignment import SpecificationDependencyAlignmentTester

# Level 4: Builder ↔ Configuration Alignment

__all__ = [
    # Level 1
    "ScriptContractAlignmentTester",
    
    # Level 2
    "ContractSpecificationAlignmentTester",
    
    # Level 3
    "SpecificationDependencyAlignmentTester",
]
