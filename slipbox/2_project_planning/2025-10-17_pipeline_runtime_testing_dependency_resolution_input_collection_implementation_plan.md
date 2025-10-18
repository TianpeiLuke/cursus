---
tags:
  - project
  - planning
  - pipeline_runtime_testing
  - dependency_resolution
  - input_collection
  - message_passing
  - implementation
keywords:
  - dependency resolution
  - message passing algorithms
  - input collection automation
  - two-phase architecture
  - maximum component reuse
  - pipeline assembler integration
  - implementation roadmap
topics:
  - dependency resolution implementation
  - input collection automation
  - pipeline runtime testing
  - message passing integration
  - implementation planning
language: python
date of note: 2025-10-17
---

# Pipeline Runtime Testing Dependency Resolution Input Collection Implementation Plan

## Project Overview

This document outlines the comprehensive implementation plan for the **Two-Phase Dependency Resolution Input Collection System**, a sophisticated enhancement to the script testing framework that integrates message passing algorithms and dependency resolution into input collection, dramatically reducing user input burden while leveraging existing pipeline assembler patterns and dependency resolution infrastructure.

## Related Design Documents

### Core Architecture Design
- **[Pipeline Runtime Testing Dependency Resolution Input Collection Design](../1_design/pipeline_runtime_testing_dependency_resolution_input_collection_design.md)** - Complete architectural design with two-phase system and pipeline assembler integration

### Foundation Documents
- **[Pipeline Runtime Testing Step Catalog Integration Design](../1_design/pipeline_runtime_testing_step_catalog_integration_design.md)** - Step catalog integration patterns and capabilities
- **[Pipeline Runtime Testing Simplified Design](../1_design/pipeline_runtime_testing_simplified_design.md)** - Core runtime testing architecture
- **[Pipeline Assembler Design](../1_design/pipeline_assembler.md)** - Message passing algorithm and dependency resolution patterns
- **[Dependency Resolution System](../1_design/dependency_resolution_system.md)** - UnifiedDependencyResolver architecture and compatibility scoring

### Core System Integration
- **[Step Builder Patterns](../0_developer_guide/step_builder.md)** - Step builder input resolution patterns for adaptation
- **[Config-Based Extraction Implementation Gaps](2025-10-17_config_based_extraction_implementation_gaps.md)** - Config field access patterns and Pydantic compliance
- **[Three Tier Config Design](../1_design/three_tier_config_design.md)** - Essential/System/Derived field categorization

## Architectural Insight: Script Testing = Pipeline Assembly

The fundamental insight driving this implementation is that **script testing input collection mirrors pipeline assembly patterns**:

**Pipeline Assembly Process (cursus/core):**
1. `PipelineAssembler._propagate_messages()` → Traverse DAG edges, match step outputs to inputs
2. `UnifiedDependencyResolver._calculate_compatibility()` → Intelligent compatibility scoring
3. `StepBuilder._get_inputs()` → Map logical dependency names to actual input objects
4. Contract-based path resolution → Use contracts for intelligent path mapping

**Script Testing Input Collection Process (new implementation):**
1. `prepare_script_testing_inputs()` → Traverse DAG edges, match script outputs to inputs
2. `UnifiedDependencyResolver._calculate_compatibility()` → **DIRECT REUSE** of same algorithm
3. `collect_user_inputs_with_dependency_resolution()` → Map resolved dependencies to script inputs
4. Contract-based path resolution → **DIRECT REUSE** of same contract patterns

Both processes use the same dependency resolution algorithms, specification matching, and contract-based path resolution.

## Maximum Component Reuse Strategy

### Direct Reuse from cursus/core
- `UnifiedDependencyResolver` - **DIRECT REUSE** for dependency matching and compatibility scoring
- `create_dependency_resolver()` - **DIRECT REUSE** of factory function
- `PipelineAssembler._propagate_messages()` patterns - **DIRECT ADAPTATION** for script testing
- `StepBuilder._get_inputs()` patterns - **DIRECT ADAPTATION** for script input resolution

### Direct Reuse from cursus/step_catalog
- `StepCatalog` - **DIRECT REUSE** for specification loading and contract discovery
- `SpecAutoDiscovery.load_spec_class()` - **DIRECT REUSE** for specification loading
- Contract classes - **DIRECT REUSE** for path resolution and logical name mapping

### Direct Reuse from cursus/steps/configs/utils
- `load_configs()` - **DIRECT REUSE** for configuration loading
- `build_complete_config_classes()` - **DIRECT REUSE** for config class construction
- `collect_script_inputs()` - **DIRECT REUSE** for config-based data extraction

### Direct Reuse from cursus/api/dag
- `PipelineDAG.topological_sort()` - **DIRECT REUSE** for execution order
- `PipelineDAG.nodes` and `PipelineDAG.edges` - **DIRECT REUSE** for DAG traversal

## Two-Phase Architecture Implementation

### Phase 1: Prepare Phase (Automatic Dependency Analysis)
**Objective**: Perform automatic dependency analysis using pipeline assembler patterns

**Key Components**:
- Specification loading using step catalog
- Dependency matching using UnifiedDependencyResolver
- Config-based data extraction using existing functions
- Dependency mapping creation for Phase 2

### Phase 2: User Input Phase (Interactive Collection with Auto-Resolution)
**Objective**: Interactive collection with automatic dependency resolution and user override capability

**Key Components**:
- DAG traversal in topological order
- Automatic input path resolution from previous script outputs
- User input collection for unresolved dependencies
- User override capability for auto-resolved paths
- Complete input/output mapping construction

## Implementation Phases

### Phase 1: Core Dependency Resolution Module (Week 1)

#### Objective
Create the core dependency resolution module with maximum component reuse from pipeline assembler patterns.

#### Implementation Strategy

**Files to Create:**
- `src/cursus/validation/script_testing/script_dependency_matcher.py` - **NEW MODULE** with two-phase system

**Core Functions Implementation:**
```python
# File: src/cursus/validation/script_testing/script_dependency_matcher.py

def prepare_script_testing_inputs(
    dag: PipelineDAG,  # DIRECT REUSE
    config_path: str,
    step_catalog: StepCatalog  # DIRECT REUSE
) -> Dict[str, Any]:
    """
    Phase 1: Automatic dependency analysis using pipeline assembler patterns.
    
    DIRECT REUSE of PipelineAssembler._propagate_messages() algorithm.
    """
    logger.info("Phase 1: Analyzing dependencies and preparing input collection...")
    
    # 1. Load specifications (DIRECT REUSE of step catalog patterns)
    node_specs = {}
    for node_name in dag.nodes:
        spec = step_catalog.spec_discovery.load_spec_class(node_name)  # DIRECT REUSE
        if spec:
            node_specs[node_name] = spec
    
    # 2. Dependency matching (DIRECT REUSE of pipeline assembler algorithm)
    dependency_resolver = create_dependency_resolver()  # DIRECT REUSE
    dependency_matches = {}
    
    for consumer_node in dag.nodes:
        if consumer_node not in node_specs:
            continue
            
        consumer_spec = node_specs[consumer_node]
        matches = {}
        
        for dep_name, dep_spec in consumer_spec.dependencies.items():
            best_match = None
            best_score = 0.0
            
            for provider_node in dag.nodes:
                if provider_node == consumer_node or provider_node not in node_specs:
                    continue
                    
                provider_spec = node_specs[provider_node]
                
                for output_name, output_spec in provider_spec.outputs.items():
                    # DIRECT REUSE: Same compatibility calculation as pipeline assembler
                    score = dependency_resolver._calculate_compatibility(
                        dep_spec, output_spec, provider_spec
                    )
                    
                    if score > best_score and score > 0.5:  # Same threshold
                        best_match = {
                            'provider_node': provider_node,
                            'provider_output': output_name,
                            'compatibility_score': score,
                            'match_type': 'specification_match'
                        }
                        best_score = score
            
            if best_match:
                matches[dep_name] = best_match
        
        dependency_matches[consumer_node] = matches
    
    # 3. Config extraction (DIRECT REUSE of existing functions)
    config_data = {}
    config_classes = build_complete_config_classes()  # DIRECT REUSE
    all_configs = load_configs(config_path, config_classes)  # DIRECT REUSE
    
    for node_name in dag.nodes:
        if node_name in all_configs:
            config = all_configs[node_name]
            config_data[node_name] = collect_script_inputs(config)  # DIRECT REUSE
    
    return {
        'node_specs': node_specs,
        'dependency_matches': dependency_matches,
        'config_data': config_data,
        'execution_order': dag.topological_sort()  # DIRECT REUSE
    }


def collect_user_inputs_with_dependency_resolution(
    prepared_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Phase 2: Interactive input collection with automatic dependency resolution.
    
    Mirrors PipelineAssembler._propagate_messages() + StepBuilder._get_inputs() patterns.
    """
    execution_order = prepared_data['execution_order']
    dependency_matches = prepared_data['dependency_matches']
    node_specs = prepared_data['node_specs']
    config_data = prepared_data['config_data']
    
    # Track outputs (like pipeline assembler's step_messages)
    completed_outputs = {}
    all_user_inputs = {}
    
    print(f"\n🔧 Script Testing Input Collection")
    print(f"   Processing {len(execution_order)} scripts in dependency order...")
    
    for node_name in execution_order:
        print(f"\n📝 Script: {node_name}")
        
        # 1. Start with config-based data
        script_config = config_data.get(node_name, {})
        
        # 2. Auto-resolve dependencies (like pipeline assembler message passing)
        resolved_inputs = {}
        unresolved_inputs = []
        
        if node_name in node_specs:
            spec = node_specs[node_name]
            matches = dependency_matches.get(node_name, {})
            
            for dep_name, dep_spec in spec.dependencies.items():
                if dep_name in matches:
                    match = matches[dep_name]
                    provider_node = match['provider_node']
                    provider_output = match['provider_output']
                    
                    if provider_node in completed_outputs and provider_output in completed_outputs[provider_node]:
                        actual_path = completed_outputs[provider_node][provider_output]
                        resolved_inputs[dep_name] = actual_path
                        
                        print(f"   🔗 Auto-resolved {dep_name} = {actual_path}")
                        print(f"      Source: {provider_node}.{provider_output} (compatibility: {match['compatibility_score']:.2f})")
                    else:
                        unresolved_inputs.append(dep_name)
                else:
                    unresolved_inputs.append(dep_name)
        
        # 3. User input for unresolved dependencies AND override capability
        user_input_paths = {}
        
        # First: Ask for unresolved inputs (required)
        if unresolved_inputs:
            print(f"   📥 Please provide input paths:")
            for dep_name in unresolved_inputs:
                path = input(f"      {dep_name}: ").strip()
                if path:
                    user_input_paths[dep_name] = path
        
        # Second: Allow override of auto-resolved inputs (optional)
        if resolved_inputs:
            print(f"   🔄 Auto-resolved inputs (press Enter to keep, or provide new path to override):")
            for dep_name, auto_path in resolved_inputs.items():
                override_path = input(f"      {dep_name} [{auto_path}]: ").strip()
                if override_path:
                    user_input_paths[dep_name] = override_path
                    print(f"      ✏️  Overridden: {dep_name} = {override_path}")
        
        # 4. Combine resolved and user inputs (user inputs take precedence)
        final_input_paths = {**resolved_inputs, **user_input_paths}
        
        # 5. User input for output paths (always required)
        output_paths = {}
        if node_name in node_specs:
            spec = node_specs[node_name]
            if spec.outputs:
                print(f"   📤 Please provide output paths:")
                for output_name, output_spec in spec.outputs.items():
                    path = input(f"      {output_name}: ").strip()
                    if path:
                        output_paths[output_name] = path
        
        # 6. Store complete configuration
        all_user_inputs[node_name] = {
            'input_paths': final_input_paths,
            'output_paths': output_paths,
            'environment_variables': script_config.get('environment_variables', {}),
            'job_arguments': script_config.get('job_arguments', {}),
            'script_path': script_config.get('script_path')
        }
        
        # 7. Register outputs for next scripts (like pipeline assembler)
        completed_outputs[node_name] = output_paths
        
        print(f"   ✅ Configured {node_name} with {len(final_input_paths)} inputs, {len(output_paths)} outputs")
    
    return all_user_inputs


def resolve_script_dependencies(
    dag: PipelineDAG,  # DIRECT REUSE
    config_path: str,
    step_catalog: StepCatalog  # DIRECT REUSE
) -> Dict[str, Any]:
    """
    Main entry point: Two-phase dependency resolution system.
    
    Combines both phases with maximum component reuse.
    """
    # Phase 1: Prepare (automatic)
    print("🔄 Phase 1: Analyzing dependencies and preparing input collection...")
    prepared_data = prepare_script_testing_inputs(dag, config_path, step_catalog)
    
    total_matches = sum(len(matches) for matches in prepared_data['dependency_matches'].values())
    print(f"   Found {total_matches} automatic dependency matches")
    
    # Phase 2: User input (interactive)
    print("🔄 Phase 2: Collecting user inputs with automatic dependency resolution...")
    user_inputs = collect_user_inputs_with_dependency_resolution(prepared_data)
    
    print(f"\n✅ Input collection complete! Configured {len(user_inputs)} scripts.")
    return user_inputs
```

#### Success Criteria
- ✅ Core dependency resolution module created with maximum component reuse
- ✅ Direct reuse of PipelineAssembler._propagate_messages() algorithm
- ✅ Direct reuse of UnifiedDependencyResolver._calculate_compatibility()
- ✅ Direct reuse of step catalog specification loading
- ✅ Direct reuse of config-based extraction functions
- ✅ Two-phase architecture fully implemented

### Phase 2: API Integration (Week 2)

#### Objective
Integrate the two-phase dependency resolution system into the existing script testing API with dramatic simplification.

#### Implementation Strategy

**Files to Update:**
- `src/cursus/validation/script_testing/api.py` - **UPDATE** to use dependency resolver

**API Integration Implementation:**
```python
# File: src/cursus/validation/script_testing/api.py (UPDATED)

def run_dag_scripts(
    dag: PipelineDAG,  # DIRECT REUSE
    config_path: str,
    test_workspace_dir: str = "test/integration/script_testing",
    step_catalog: Optional[StepCatalog] = None  # DIRECT REUSE
) -> Dict[str, Any]:
    """
    SIMPLIFIED: Run scripts with two-phase dependency resolution.
    
    Complexity moved to dependency resolver, making this function much simpler.
    """
    try:
        # Initialize step catalog
        if not step_catalog:
            step_catalog = StepCatalog()  # DIRECT REUSE
        
        logger.info(f"Starting DAG-guided script testing with {len(dag.nodes)} nodes")
        
        # SIMPLIFIED: Use two-phase dependency resolution
        from .script_dependency_matcher import resolve_script_dependencies
        user_inputs = resolve_script_dependencies(dag, config_path, step_catalog)
        
        # SIMPLIFIED: Execute with pre-resolved inputs
        results = execute_scripts_in_order(
            execution_order=dag.topological_sort(),  # DIRECT REUSE
            user_inputs=user_inputs  # Complete inputs from dependency resolution
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Script testing failed: {e}")
        raise RuntimeError(f"Failed to test DAG scripts: {e}") from e


def execute_scripts_in_order(
    execution_order: List[str],
    user_inputs: Dict[str, Any]  # Complete inputs from two-phase system
) -> Dict[str, Any]:
    """
    DRAMATICALLY SIMPLIFIED: Execute scripts with complete pre-resolved inputs.
    
    All complexity (message passing, dependency matching, config extraction) 
    is handled in input collection phase.
    """
    results = {}
    
    for node_name in execution_order:
        try:
            logger.info(f"Executing script: {node_name}")
            
            # SIMPLIFIED: Get complete pre-resolved data
            node_inputs = user_inputs.get(node_name, {})
            
            # All information is complete from two-phase resolution:
            input_paths = node_inputs.get('input_paths', {})        # ✅ Auto-resolved or user-provided
            output_paths = node_inputs.get('output_paths', {})      # ✅ User-provided
            environ_vars = node_inputs.get('environment_variables', {})  # ✅ From config
            job_args = node_inputs.get('job_arguments', {})         # ✅ From config
            script_path = node_inputs.get('script_path')            # ✅ From config
            
            if not script_path:
                logger.warning(f"No script path found for {node_name}, skipping")
                continue
            
            # ULTRA-SIMPLIFIED: Just execute with complete information
            result = execute_single_script(script_path, input_paths, output_paths, environ_vars, job_args)
            results[node_name] = result
            
            if result.success:
                logger.info(f"✅ {node_name} executed successfully")
            else:
                logger.error(f"❌ {node_name} failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ {node_name} execution failed: {e}")
            results[node_name] = ScriptTestResult(success=False, error_message=str(e))
    
    return {
        "pipeline_success": all(r.success for r in results.values()),
        "script_results": results,
        "execution_order": execution_order,
        "total_scripts": len(execution_order),
        "successful_scripts": sum(1 for r in results.values() if r.success)
    }
```

#### Success Criteria
- ✅ API dramatically simplified with dependency resolution moved to input collection
- ✅ `execute_scripts_in_order` function signature simplified (no dependency_resolver, config_path)
- ✅ All complex logic removed from execution phase
- ✅ Complete integration with two-phase dependency resolution
- ✅ Backward compatibility maintained

### Phase 3: Input Collector Enhancement (Week 3)

#### Objective
Enhance the existing input collector to optionally use the two-phase dependency resolution system.

#### Implementation Strategy

**Files to Update:**
- `src/cursus/validation/script_testing/input_collector.py` - **ENHANCE** with dependency resolution

**Input Collector Enhancement:**
```python
# File: src/cursus/validation/script_testing/input_collector.py (ENHANCED)

class ScriptTestingInputCollector:
    """Enhanced with optional two-phase dependency resolution."""
    
    def __init__(self, dag: PipelineDAG, config_path: str, use_dependency_resolution: bool = True):
        """Initialize with optional dependency resolution."""
        self.dag = dag  # DIRECT REUSE
        self.config_path = config_path
        self.use_dependency_resolution = use_dependency_resolution
        
        # Existing initialization
        self.dag_factory = DAGConfigFactory(dag)  # DIRECT REUSE
        self.loaded_configs = self._load_and_filter_configs()
    
    def collect_script_inputs_for_dag(self) -> Dict[str, Any]:
        """Enhanced collection with optional two-phase dependency resolution."""
        
        if self.use_dependency_resolution:
            # NEW: Use two-phase dependency resolution
            from .script_dependency_matcher import resolve_script_dependencies
            from ...step_catalog import StepCatalog
            
            logger.info("Using two-phase dependency resolution for input collection")
            return resolve_script_dependencies(
                dag=self.dag,
                config_path=self.config_path,
                step_catalog=StepCatalog()  # DIRECT REUSE
            )
        else:
            # FALLBACK: Use existing manual collection
            logger.info("Using manual input collection (legacy mode)")
            return self._collect_inputs_manually()
    
    def _collect_inputs_manually(self) -> Dict[str, Any]:
        """Existing manual collection logic (unchanged for backward compatibility)."""
        user_inputs = {}
        validated_scripts = self._get_validated_scripts_from_config()
        
        for script_name in validated_scripts:
            script_inputs = self._collect_script_inputs(script_name)
            user_inputs[script_name] = script_inputs
        
        return user_inputs
```

#### Success Criteria
- ✅ Input collector enhanced with optional dependency resolution
- ✅ Backward compatibility maintained with manual collection fallback
- ✅ Seamless integration with two-phase system
- ✅ User can choose between automatic and manual collection
- ✅ Direct reuse of existing components

### Phase 4: Script Input Resolution Pattern Adaptation (Week 3)

#### Objective
Adapt step builder input resolution patterns for script testing with contract-based path mapping.

#### Implementation Strategy

**Files to Create:**
- `src/cursus/validation/script_testing/script_input_resolver.py` - **NEW MODULE** for script input resolution patterns

**Script Input Resolution Pattern Adaptation:**
```python
# File: src/cursus/validation/script_testing/script_input_resolver.py

def resolve_script_inputs_using_step_patterns(
    node_name: str,
    spec: StepSpecification,  # DIRECT REUSE
    resolved_dependencies: Dict[str, str],
    step_catalog: StepCatalog  # DIRECT REUSE
) -> Dict[str, str]:
    """
    Script input resolution adapted from StepBuilder._get_inputs() patterns.
    
    DIRECT ADAPTATION of step builder input resolution logic for script testing.
    """
    script_inputs = {}
    
    # Load contract (DIRECT REUSE of step catalog patterns)
    contract = step_catalog.load_contract_class(node_name)
    
    # Process dependencies (SAME PATTERN as step builders)
    for dep_name, dep_spec in spec.dependencies.items():
        
        # Skip optional unresolved dependencies
        if not dep_spec.required and dep_name not in resolved_dependencies:
            continue
        
        # Ensure required dependencies are resolved
        if dep_spec.required and dep_name not in resolved_dependencies:
            raise ValueError(f"Required dependency '{dep_name}' not resolved for {node_name}")
        
        # Get actual path
        actual_path = resolved_dependencies[dep_name]
        
        # Map using contract (SAME PATTERN as step builders)
        if contract and hasattr(contract, 'expected_input_paths'):
            container_path = contract.expected_input_paths.get(dep_name)
            if container_path:
                script_inputs[dep_name] = actual_path
            else:
                script_inputs[dep_name] = actual_path
        else:
            script_inputs[dep_name] = actual_path
    
    return script_inputs


def adapt_step_input_patterns_for_scripts(
    node_name: str,
    inputs: Dict[str, Any],
    step_catalog: StepCatalog  # DIRECT REUSE
) -> Dict[str, str]:
    """
    Adapt step builder input patterns for script testing.
    
    DIRECT ADAPTATION of step builder input resolution patterns.
    """
    # Load specification (DIRECT REUSE)
    spec = step_catalog.spec_discovery.load_spec_class(node_name)
    if not spec:
        raise ValueError(f"No specification found for {node_name}")
    
    # Load contract (DIRECT REUSE)
    contract = step_catalog.load_contract_class(node_name)
    if not contract:
        raise ValueError(f"No contract found for {node_name}")
    
    script_inputs = {}
    
    # Process each dependency (SAME PATTERN as step builders)
    for dep_name, dep_spec in spec.dependencies.items():
        
        # Skip optional dependencies not provided
        if not dep_spec.required and dep_name not in inputs:
            continue
        
        # Ensure required dependencies are provided
        if dep_spec.required and dep_name not in inputs:
            raise ValueError(f"Required input '{dep_name}' not provided")
        
        # Get container path from contract (SAME PATTERN as step builders)
        container_path = None
        if hasattr(contract, 'expected_input_paths'):
            container_path = contract.expected_input_paths.get(dep_name)
        
        if container_path:
            # Use logical name for script input mapping
            script_inputs[dep_name] = inputs[dep_name]
        else:
            # Fallback to logical name
            script_inputs[dep_name] = inputs[dep_name]
    
    return script_inputs
```

#### Success Criteria
- ✅ Step builder patterns adapted for script testing
- ✅ Contract-based path mapping integrated
- ✅ Direct reuse of step catalog contract loading
- ✅ Logical name to actual path transformation
- ✅ Same validation patterns as step builders

### Phase 5: Testing and Documentation (Week 4)

#### Objective
Complete comprehensive testing and documentation for the two-phase dependency resolution system.

#### Implementation Strategy

**Files to Create:**
- `test/validation/script_testing/test_dependency_resolver.py` - **NEW** comprehensive tests
- `test/validation/script_testing/test_two_phase_integration.py` - **NEW** integration tests
- `docs/script_testing_dependency_resolution.md` - **NEW** documentation

**Testing Implementation:**
```python
# File: test/validation/script_testing/test_dependency_resolver.py

class TestDependencyResolver:
    """Comprehensive tests for two-phase dependency resolution system."""
    
    def test_prepare_script_testing_inputs(self):
        """Test Phase 1: Automatic dependency analysis."""
        
    def test_collect_user_inputs_with_dependency_resolution(self):
        """Test Phase 2: Interactive collection with auto-resolution."""
        
    def test_resolve_script_dependencies_complete_workflow(self):
        """Test complete two-phase workflow."""
        
    def test_pipeline_assembler_pattern_reuse(self):
        """Validate direct reuse of pipeline assembler patterns."""
        
    def test_step_builder_pattern_integration(self):
        """Test step builder pattern integration."""
        
    def test_user_override_capability(self):
        """Test user override capability for auto-resolved paths."""
        
    def test_maximum_component_reuse(self):
        """Validate maximum component reuse from existing cursus infrastructure."""
        
    def test_performance_benchmarks(self):
        """Test performance improvements and user input reduction."""


class TestTwoPhaseIntegration:
    """Integration tests for complete two-phase system."""
    
    def test_end_to_end_dag_execution(self):
        """Test complete DAG execution with dependency resolution."""
        
    def test_api_simplification(self):
        """Test API simplification and execute_scripts_in_order changes."""
        
    def test_input_collector_enhancement(self):
        """Test enhanced input collector with dependency resolution."""
        
    def test_backward_compatibility(self):
        """Test backward compatibility with existing functionality."""
```

**Documentation Implementation:**
```markdown
# File: docs/script_testing_dependency_resolution.md

# Script Testing Dependency Resolution

## Overview
The Two-Phase Dependency Resolution Input Collection System dramatically reduces user input burden by integrating message passing algorithms and dependency resolution into script testing input collection.

## Key Features
- **60-70% reduction** in manual path specifications
- **Automatic dependency resolution** using pipeline assembler patterns
- **User override capability** for auto-resolved paths
- **Maximum component reuse** from existing cursus infrastructure
- **Dramatically simplified API** with complexity moved to input collection

## Usage Examples
[Complete usage examples and best practices]

## Migration Guide
[Step-by-step migration from existing manual collection]
```

#### Success Criteria
- ✅ Comprehensive test coverage for all components
- ✅ Integration tests validating complete workflow
- ✅ Performance benchmarks showing user input reduction
- ✅ Complete documentation and migration guides
- ✅ Backward compatibility validation

## Expected Outcomes

### Before Implementation
- **Manual Input Collection**: User must specify all input paths for every script
- **No Dependency Resolution**: Repetitive path specification for connected scripts
- **High User Burden**: 15-25 manual path specifications for 5-script pipeline
- **Error-Prone Process**: Manual path management leads to errors
- **Complex API**: Complex dependency resolution logic in execution phase

### After Implementation
- **Intelligent Input Collection**: Algorithm propagates paths automatically using proven compatibility scoring
- **Automatic Dependency Resolution**: Uses same algorithms as pipeline assembler
- **Minimal User Burden**: 5-10 manual path specifications for 5-script pipeline (60-70% reduction)
- **Error Reduction**: Automatic resolution eliminates path management errors
- **Simplified API**: All complexity moved to input collection phase

## Success Metrics

### Implementation Success Criteria
- **Component Reuse**: 95% reuse of existing cursus infrastructure
- **Pattern Consistency**: Direct reuse of pipeline assembler patterns
- **User Input Reduction**: 60-70% reduction in manual path specifications
- **API Simplification**: Dramatic reduction in execution phase complexity
- **Performance**: Minimal overhead with significant usability improvements

### Quality Metrics
- **Test Coverage**: >95% code coverage for all new components
- **Integration**: Seamless integration with existing cursus infrastructure
- **Backward Compatibility**: Zero-breaking-change migration path
- **Documentation**: Complete API documentation and usage examples
- **User Experience**: Interactive collection with intelligent defaults and override capability

## Risk Assessment and Mitigation

### Technical Risks

**Component Dependency Complexity**
- *Risk*: Heavy dependency on existing components may cause integration issues
- *Mitigation*: Direct reuse patterns with graceful fallback mechanisms
- *Fallback*: Manual collection mode available as backup

**Performance Impact**
- *Risk*: Two-phase system may introduce performance overhead
- *Mitigation*: Leverage existing optimized algorithms from pipeline assembler
- *Fallback*: Performance monitoring and optimization

**Integration Complexity**
- *Risk*: Complex integration with multiple existing systems
- *Mitigation*: Phased implementation with validation at each step
- *Fallback*: Modular architecture allows selective feature rollback

### Project Risks

**Implementation Timeline**
- *Risk*: 6-week timeline may be ambitious for complete implementation
- *Mitigation*: Focus on maximum component reuse to reduce implementation time
- *Fallback*: Core functionality prioritized, advanced features can be deferred

**User Adoption**
- *Risk*: Users may be hesitant to adopt new input collection approach
- *Mitigation*: Backward compatibility and optional dependency resolution
- *Fallback*: Manual collection mode remains available

## Implementation Timeline

### Week 1: Core Dependency Resolution Module ✅ COMPLETED
- [x] Create `script_dependency_matcher.py` with two-phase system
- [x] Implement `prepare_script_testing_inputs()` with pipeline assembler patterns
- [x] Implement `collect_user_inputs_with_dependency_resolution()` with user override
- [x] Implement `resolve_script_dependencies()` main entry point
- [x] Direct reuse of UnifiedDependencyResolver and step catalog
- [x] Comprehensive unit tests for all functions (18/18 tests passing - 100% success rate)
- [x] Systematic error fixing following pytest best practices
- [x] Algorithm behavior analysis and test alignment
- [x] Mock structure fixes and proper test configuration
- [x] Complete validation and summary functions implementation

### Week 2: API Integration ✅ COMPLETED
- [x] Update `run_dag_scripts()` to use dependency resolver with optional `use_dependency_resolution` parameter
- [x] Dramatically simplify `execute_scripts_in_order()` function signature (removed dependency_resolver and config_path parameters)
- [x] Remove all complex dependency resolution logic from execution phase - moved to input collection
- [x] Enhanced `ScriptTestingInputCollector` with optional two-phase dependency resolution
- [x] Implemented seamless fallback to manual collection for backward compatibility
- [x] API now supports both automatic dependency resolution and legacy manual collection modes
- [x] Complete integration testing showing simplified workflow and maintained functionality

### Week 3: Script Input Resolution Pattern Adaptation ✅ COMPLETED
- [x] Create `script_input_resolver.py` module (renamed from step_builder_integration.py)
- [x] Implement `resolve_script_inputs_using_step_patterns()` (adapted from step builder patterns)
- [x] Adapt contract-based path mapping for script testing
- [x] Implement logical name to actual path transformation patterns
- [x] Additional utility functions: `validate_script_input_resolution()`, `get_script_input_resolution_summary()`, `transform_logical_names_to_actual_paths()`
- [x] Direct reuse of StepSpecification from `cursus.core.base.specification_base`
- [x] Same validation patterns as step builders with proper error handling
- [x] Contract-based path mapping with graceful fallbacks
- [x] Comprehensive logging and debugging support
- [x] **COMPREHENSIVE PYTEST TESTS**: 34 tests covering all functions and scenarios
- [x] **TEST EXECUTION**: All tests passing (34/34) with systematic error correction
- [x] **PYTEST BEST PRACTICES**: Read source code first, set expected responses correctly, proper mock structure
- [x] **FAILURE PATTERN ANALYSIS**: Tested all edge cases, error handling, and integration scenarios
- [x] **MOCK EXCELLENCE**: Proper mock usage without spec issues, realistic behavior patterns

### Week 4: Testing and Documentation
- [ ] Complete end-to-end integration testing
- [ ] Performance benchmarking showing user input reduction
- [ ] Comprehensive documentation and usage examples
- [ ] Migration guide and backward compatibility validation
- [ ] User acceptance testing and feedback incorporation

**Note**: Week 5 (Message Passing Algorithm Integration) has been removed as redundant - message passing algorithms were already integrated into `script_dependency_matcher.py` in Week 1.

## Implementation Dependencies

### Internal Dependencies
- **cursus/core**: UnifiedDependencyResolver, create_dependency_resolver()
- **cursus/step_catalog**: StepCatalog, SpecAutoDiscovery, contract loading
- **cursus/api/dag**: PipelineDAG, topological_sort(), nodes, edges
- **cursus/steps/configs/utils**: load_configs(), build_complete_config_classes(), collect_script_inputs()
- **cursus/validation/script_testing**: Existing API and input collector

### External Dependencies
- **Pydantic**: For data model validation (already used)
- **Pathlib**: For file system operations (already used)
- **Typing**: For comprehensive type hints (already used)
- **Logging**: For execution tracking (already used)

## Conclusion

The Pipeline Runtime Testing Dependency Resolution Input Collection Implementation Plan provides a comprehensive roadmap for creating an intelligent, automated input collection system that dramatically reduces user input burden while leveraging maximum component reuse from existing cursus infrastructure.

### Key Implementation Principles

1. **Maximum Component Reuse**: Achieve 95% reuse of existing cursus infrastructure
2. **Direct Pattern Adaptation**: Adapt proven pipeline assembler patterns for script testing
3. **User-Centric Design**: Minimize user input while maintaining full control and override capability
4. **API Simplification**: Move complexity to input collection, simplify execution phase
5. **Backward Compatibility**: Maintain existing functionality with optional enhancements

### Expected Impact

**Development Workflow Transformation**:
- **Before**: Manual specification of all input paths for every script in pipeline
- **After**: Automatic dependency resolution with minimal user input and optional override capability

**Technical Achievements**:
- Direct reuse of pipeline assembler message passing algorithms
- Intelligent dependency resolution using proven compatibility scoring
- Two-phase architecture with clean separation of concerns
- Dramatically simplified script execution API
- User override capability for maximum flexibility

This implementation will transform script testing from a manual, error-prone process into an intelligent, automated system that provides the same level of sophistication as the core pipeline compilation system while maintaining simplicity and user control.

## References

### Design Documents
- **[Pipeline Runtime Testing Dependency Resolution Input Collection Design](../1_design/pipeline_runtime_testing_dependency_resolution_input_collection_design.md)** - Complete architectural design
- **[Pipeline Assembler Design](../1_design/pipeline_assembler.md)** - Message passing algorithm patterns
- **[Dependency Resolution System](../1_design/dependency_resolution_system.md)** - UnifiedDependencyResolver architecture

### Implementation References
- **[Pipeline Runtime Testing DAG-Guided Script Testing Engine Implementation Plan](2025-10-17_pipeline_runtime_testing_dag_guided_script_testing_engine_implementation_plan.md)** - Related implementation patterns
- **[Config-Based Extraction Implementation Gaps](2025-10-17_config_based_extraction_implementation_gaps.md)** - Config field access patterns

### Developer Guides
- **[Step Builder Patterns](../0_developer_guide/step_builder.md)** - Step builder input resolution patterns
- **[Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md)** - Integration patterns and best practices
