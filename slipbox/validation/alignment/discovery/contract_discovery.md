---
tags:
  - code
  - validation
  - alignment
  - contract_discovery
  - discovery_engine
keywords:
  - ContractDiscoveryEngine
  - contract discovery
  - contract mapping
  - entry point mapping
  - contract references
  - script matching
topics:
  - alignment validation
  - contract discovery
  - file mapping
language: python
date of note: 2025-09-07
---

# Contract Discovery Engine

Engine for discovering and mapping contract files in the alignment validation system.

## Overview

The Contract Discovery Engine provides robust contract file discovery using multiple strategies including entry point mapping, specification file contract references, and naming convention patterns. It handles discovery and mapping of contract files to their corresponding scripts, ensuring only valid contracts are processed during validation.

The engine supports dynamic module loading with proper path management, flexible contract matching strategies, and comprehensive error handling for malformed contracts or missing dependencies.

## Classes and Methods

### Classes
- [`ContractDiscoveryEngine`](#contractdiscoveryengine) - Main engine for discovering and mapping contract files

## API Reference

### ContractDiscoveryEngine

_class_ cursus.validation.alignment.discovery.contract_discovery.ContractDiscoveryEngine(_contracts_dir_)

Engine for discovering and mapping contract files using multiple discovery strategies.

**Parameters:**
- **contracts_dir** (_str_) – Directory containing script contracts

```python
from cursus.validation.alignment.discovery.contract_discovery import ContractDiscoveryEngine

discovery = ContractDiscoveryEngine('src/cursus/steps/contracts')
```

#### discover_all_contracts

discover_all_contracts()

Discover all contract files in the contracts directory using naming convention patterns.

**Returns:**
- **List[str]** – Sorted list of contract names (without '_contract' suffix)

```python
all_contracts = discovery.discover_all_contracts()
print(f"Found contracts: {all_contracts}")
```

#### discover_contracts_with_scripts

discover_contracts_with_scripts()

Discover contracts that have corresponding scripts by checking their entry_point field. This method loads each contract and verifies that the script file referenced in the entry_point field actually exists, preventing validation errors for contracts without corresponding scripts.

**Returns:**
- **List[str]** – Sorted list of contract names that have corresponding scripts

```python
valid_contracts = discovery.discover_contracts_with_scripts()
print(f"Valid contracts: {valid_contracts}")
```

#### build_entry_point_mapping

build_entry_point_mapping()

Build a mapping from entry_point values to contract file names. Results are cached for performance.

**Returns:**
- **Dict[str, str]** – Dictionary mapping entry_point (script filename) to contract filename

```python
mapping = discovery.build_entry_point_mapping()
print(f"Entry point mapping: {mapping}")
# Example: {'preprocessing.py': 'preprocessing_contract.py'}
```

#### extract_script_contract_from_spec

extract_script_contract_from_spec(_spec_file_)

Extract the script_contract field from a specification file using dynamic module loading.

**Parameters:**
- **spec_file** (_Path_) – Path to the specification file

**Returns:**
- **Optional[str]** – Script name from contract entry_point, or None if not found

```python
from pathlib import Path

spec_file = Path('src/cursus/steps/specifications/preprocessing_spec.py')
contract_ref = discovery.extract_script_contract_from_spec(spec_file)
print(f"Contract reference: {contract_ref}")
```

#### extract_contract_reference_from_spec

extract_contract_reference_from_spec(_spec_file_)

Extract contract references from specification files using regex patterns as fallback method.

**Parameters:**
- **spec_file** (_Path_) – Path to the specification file

**Returns:**
- **Optional[str]** – Contract name from import statements, or None if not found

```python
contract_ref = discovery.extract_contract_reference_from_spec(spec_file)
```

#### contracts_match

contracts_match(_contract_from_spec_, _target_contract_name_)

Determine if contracts match using flexible matching strategies including direct match, extension handling, and prefix matching.

**Parameters:**
- **contract_from_spec** (_str_) – Contract name extracted from specification
- **target_contract_name** (_str_) – Target contract name to match against

**Returns:**
- **bool** – True if contracts match using any matching strategy

```python
matches = discovery.contracts_match('preprocessing', 'preprocessing_contract')
print(f"Contracts match: {matches}")
```

## Related Documentation

- [Unified Alignment Tester](../unified_alignment_tester.md) - Main alignment validation system
- [Contract Loader](../loaders/contract_loader.md) - Contract loading utilities
- [File Resolver](../patterns/file_resolver.md) - File resolution patterns
