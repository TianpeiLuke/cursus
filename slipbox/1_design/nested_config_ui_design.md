---
tags:
  - design
  - ui
  - nested
  - pydantic
  - config_ui
  - hierarchical
keywords:
  - nested configuration
  - hierarchical forms
  - pydantic models
  - dynamic lists
  - field arrays
  - form builder
topics:
  - nested configuration ui design
  - hierarchical pydantic structures
  - dynamic form generation
  - robust nested patterns
language: python, javascript
date of note: 2025-10-09
---

# Nested Configuration UI Design

## Problem Statement

Our current config UI system handles flat Pydantic models well, but struggles with nested structures that are common in complex configurations:

**Examples of Nested Structures:**
- **CradleDataLoadingConfig**: Contains `DataSourcesSpecificationConfig` → `List[DataSourceConfig]` → `MdsDataSourceConfig`/`EdxDataSourceConfig`/`AndesDataSourceConfig`
- **PyTorchTrainingConfig**: Contains `ModelHyperparameters` with 20+ nested fields
- **General Pattern**: Any Pydantic model with nested BaseModel fields, List[BaseModel] fields, or Union types

**Current Issues:**
1. **Inconsistent UX**: Some configs get comprehensive UI, others get basic nested object displays
2. **Not Scalable**: Manual field definitions required for each nested structure
3. **Loss of Structure**: Flattening loses logical grouping that nested structures provide
4. **Maintenance Burden**: Changes to nested structures require updates in multiple places

## Research-Based Solution: Hierarchical Configuration System

Based on research into established form builders (React Hook Form, Formily, SurveyJS), here's a robust architecture for handling nested Pydantic configurations.

### Core Architecture Components

#### 1. **Nested Structure Discovery Engine**

```python
@dataclass
class NestedStructure:
    """Represents the complete nested structure of a Pydantic model."""
    root_fields: List[FieldInfo]
    nested_models: Dict[str, NestedModelInfo]
    dynamic_lists: Dict[str, ListFieldInfo] 
    conditional_fields: Dict[str, ConditionalFieldInfo]
    validation_dependencies: List[ValidationDependency]

@dataclass
class NestedModelInfo:
    """Information about a nested Pydantic model field."""
    field_name: str
    model_class: Type[BaseModel]
    is_required: bool
    nesting_level: int
    parent_class: Type[BaseModel]
    inheritance_chain: List[str]

@dataclass
class ListFieldInfo:
    """Information about List[BaseModel] fields for dynamic arrays."""
    field_name: str
    item_model_class: Type[BaseModel]
    min_items: int
    max_items: Optional[int]
    default_items: List[Dict[str, Any]]
    supports_add_remove: bool

class NestedStructureAnalyzer:
    """Analyzes Pydantic models to discover nested structure patterns."""
    
    def analyze_model(self, config_class: Type[BaseModel]) -> NestedStructure:
        """
        Recursively analyze a Pydantic model to discover:
        - Nested BaseModel fields
        - List[BaseModel] fields (dynamic arrays)
        - Optional[BaseModel] fields (conditional sections)
        - Union types with BaseModel variants
        """
        return NestedStructure(
            root_fields=self._get_direct_fields(config_class),
            nested_models=self._discover_nested_models(config_class),
            dynamic_lists=self._discover_list_fields(config_class),
            conditional_fields=self._discover_conditional_fields(config_class),
            validation_dependencies=self._analyze_validators(config_class)
        )
    
    def _discover_nested_models(self, config_class) -> Dict[str, NestedModelInfo]:
        """Find fields that are themselves Pydantic models."""
        nested = {}
        for field_name, field_info in config_class.model_fields.items():
            if self._is_pydantic_model(field_info.annotation):
                nested[field_name] = NestedModelInfo(
                    field_name=field_name,
                    model_class=field_info.annotation,
                    is_required=field_info.is_required(),
                    nesting_level=1,
                    parent_class=config_class,
                    inheritance_chain=self._get_inheritance_chain(field_info.annotation)
                )
        return nested
    
    def _discover_list_fields(self, config_class) -> Dict[str, ListFieldInfo]:
        """Find List[BaseModel] fields for dynamic arrays."""
        list_fields = {}
        for field_name, field_info in config_class.model_fields.items():
            if self._is_list_of_models(field_info.annotation):
                item_type = self._get_list_item_type(field_info.annotation)
                list_fields[field_name] = ListFieldInfo(
                    field_name=field_name,
                    item_model_class=item_type,
                    min_items=getattr(field_info, 'min_length', 0),
                    max_items=getattr(field_info, 'max_length', None),
                    default_items=self._get_default_list_items(field_info),
                    supports_add_remove=True
                )
        return list_fields
```

#### 2. **Hierarchical Widget System**

```python
class HierarchicalConfigWidget:
    """Multi-level widget system for nested configurations."""
    
    def __init__(self, config_class: Type[BaseModel], base_config: Optional[BaseModel] = None):
        self.config_class = config_class
        self.base_config = base_config
        
        # Analyze the nested structure
        self.analyzer = NestedStructureAnalyzer()
        self.structure = self.analyzer.analyze_model(config_class)
        
        # Build widget tree
        self.widget_tree = self._build_widget_tree()
        self.field_registry = {}
        self.state_manager = NestedStateManager()
        
    def _build_widget_tree(self) -> WidgetNode:
        """Build a tree of widgets representing the nested structure."""
        root = WidgetNode(
            widget_type="root",
            config_class=self.config_class,
            fields=self.structure.root_fields,
            level=0
        )
        
        # Add nested model widgets
        for field_name, nested_info in self.structure.nested_models.items():
            nested_widget = self._create_nested_widget(nested_info)
            root.add_child(field_name, nested_widget)
        
        # Add dynamic list widgets  
        for field_name, list_info in self.structure.dynamic_lists.items():
            list_widget = self._create_list_widget(list_info)
            root.add_child(field_name, list_widget)
            
        return root
    
    def _create_nested_widget(self, nested_info: NestedModelInfo) -> WidgetNode:
        """Create a widget for a nested Pydantic model."""
        # Recursively analyze the nested model
        nested_structure = self.analyzer.analyze_model(nested_info.model_class)
        
        return WidgetNode(
            widget_type="nested_model",
            config_class=nested_info.model_class,
            fields=nested_structure.root_fields,
            is_required=nested_info.is_required,
            level=nested_info.nesting_level,
            children=self._build_children_widgets(nested_structure),
            display_mode="expandable_section"  # or "inline", "modal", "tab"
        )
    
    def _create_list_widget(self, list_info: ListFieldInfo) -> WidgetNode:
        """Create a dynamic list widget for List[BaseModel] fields."""
        # Analyze the item model structure
        item_structure = self.analyzer.analyze_model(list_info.item_model_class)
        
        return WidgetNode(
            widget_type="dynamic_list",
            config_class=list_info.item_model_class,
            is_list=True,
            min_items=list_info.min_items,
            max_items=list_info.max_items,
            item_template=self._create_list_item_template(item_structure),
            supports_add_remove=list_info.supports_add_remove,
            display_mode="accordion_list"  # or "table", "cards"
        )
```

#### 3. **Smart Field Categorization with Inheritance**

```python
class HierarchicalFieldCategorizer:
    """Categorizes fields at each nesting level with inheritance awareness."""
    
    def categorize_hierarchical_fields(self, 
                                     widget_node: WidgetNode,
                                     parent_values: Dict[str, Any] = None) -> CategorizedFields:
        """
        Categorize fields with hierarchy and inheritance awareness:
        - Tier 1 (Essential): Required fields unique to this level
        - Tier 2 (System): Optional fields with defaults at this level  
        - Tier 3 (Inherited): Fields inherited from parent levels
        - Tier 4 (Derived): Computed fields (hidden from UI)
        """
        
        # Get base categorization using existing method
        if hasattr(widget_node.config_class, 'categorize_fields'):
            base_categories = widget_node.config_class().categorize_fields()
        else:
            base_categories = self._manual_categorization(widget_node.config_class)
        
        # Enhance with inheritance information
        enhanced_categories = CategorizedFields(
            essential=[],
            system=[],
            inherited=[],
            derived=base_categories.get('derived', [])
        )
        
        # Apply inheritance logic
        for field_name in base_categories.get('essential', []):
            if parent_values and field_name in parent_values:
                # Field is inherited from parent
                enhanced_categories.inherited.append(FieldInfo(
                    name=field_name,
                    tier='inherited',
                    inherited_from=self._find_inheritance_source(field_name, parent_values),
                    default_value=parent_values[field_name],
                    can_override=True,
                    nesting_level=widget_node.level
                ))
            else:
                # Field is essential at this level
                enhanced_categories.essential.append(FieldInfo(
                    name=field_name,
                    tier='essential',
                    nesting_level=widget_node.level,
                    is_nested_field=widget_node.level > 0
                ))
        
        # Similar logic for system fields...
        for field_name in base_categories.get('system', []):
            if parent_values and field_name in parent_values:
                enhanced_categories.inherited.append(FieldInfo(
                    name=field_name,
                    tier='inherited',
                    inherited_from=self._find_inheritance_source(field_name, parent_values),
                    default_value=parent_values[field_name],
                    can_override=True,
                    nesting_level=widget_node.level
                ))
            else:
                enhanced_categories.system.append(FieldInfo(
                    name=field_name,
                    tier='system',
                    nesting_level=widget_node.level,
                    is_nested_field=widget_node.level > 0
                ))
        
        return enhanced_categories
```

#### 4. **Dynamic List Management**

```python
class DynamicListManager:
    """Manages dynamic lists of nested configurations (List[BaseModel] fields)."""
    
    def __init__(self, list_info: ListFieldInfo):
        self.list_info = list_info
        self.items = []
        self.item_widgets = {}
        
    def add_item(self, initial_data: Dict[str, Any] = None) -> str:
        """Add a new item to the list."""
        if self.list_info.max_items and len(self.items) >= self.list_info.max_items:
            raise ValueError(f"Cannot add more than {self.list_info.max_items} items")
        
        item_id = f"item_{len(self.items)}"
        
        # Create widget for the new item
        item_widget = HierarchicalConfigWidget(
            config_class=self.list_info.item_model_class,
            base_config=None
        )
        
        # Pre-populate if initial data provided
        if initial_data:
            item_widget.set_values(initial_data)
        
        self.items.append(item_id)
        self.item_widgets[item_id] = item_widget
        
        return item_id
    
    def remove_item(self, item_id: str):
        """Remove an item from the list."""
        if len(self.items) <= self.list_info.min_items:
            raise ValueError(f"Cannot have fewer than {self.list_info.min_items} items")
        
        if item_id in self.items:
            self.items.remove(item_id)
            del self.item_widgets[item_id]
    
    def get_all_values(self) -> List[Dict[str, Any]]:
        """Get values from all items in the list."""
        return [
            self.item_widgets[item_id].get_values()
            for item_id in self.items
        ]
    
    def validate_all_items(self) -> List[ValidationResult]:
        """Validate all items in the list."""
        results = []
        for item_id in self.items:
            result = self.item_widgets[item_id].validate()
            results.append(result)
        return results
```

#### 5. **Conditional Field Display**

```python
class ConditionalFieldManager:
    """Manages conditional field display based on other field values."""
    
    def __init__(self, conditional_fields: Dict[str, ConditionalFieldInfo]):
        self.conditional_fields = conditional_fields
        self.field_dependencies = self._build_dependency_graph()
        
    def should_show_field(self, field_name: str, current_values: Dict[str, Any]) -> bool:
        """Determine if a conditional field should be shown."""
        if field_name not in self.conditional_fields:
            return True  # Always show non-conditional fields
        
        condition_info = self.conditional_fields[field_name]
        
        # Evaluate condition based on current values
        return self._evaluate_condition(condition_info.condition, current_values)
    
    def _evaluate_condition(self, condition: FieldCondition, values: Dict[str, Any]) -> bool:
        """Evaluate a field condition against current values."""
        if condition.type == "equals":
            return values.get(condition.field) == condition.value
        elif condition.type == "not_equals":
            return values.get(condition.field) != condition.value
        elif condition.type == "in":
            return values.get(condition.field) in condition.value
        elif condition.type == "not_empty":
            field_value = values.get(condition.field)
            return field_value is not None and field_value != ""
        # Add more condition types as needed
        
        return True
```

### User Experience Patterns

#### 1. **Expandable Sections for Nested Models**

```
┌─────────────────────────────────────────────────────────────┐
│ 🎯 PyTorchTrainingConfig                                   │
│                                                             │
│ ⚙️ Training Configuration (Tier 2)                        │
│ ├─ training_instance_type: ml.g5.12xlarge                  │
│ ├─ training_instance_count: 1                              │
│ └─ framework_version: 1.12.0                               │
│                                                             │
│ ▼ 🤖 Model Hyperparameters (Click to expand)              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 💾 Inherited Fields (Tier 3)                           │ │
│ │ ├─ author: lukexie (from BasePipelineConfig)           │ │
│ │ ├─ bucket: my-bucket (from BasePipelineConfig)         │ │
│ │                                                         │ │
│ │ 🔥 Essential User Inputs (Tier 1)                      │ │
│ │ ├─ full_field_list: [objectId, transactionDate, ...]  │ │
│ │ ├─ cat_field_list: [category, type]                    │ │
│ │ ├─ label_name: is_abuse                                 │ │
│ │                                                         │ │
│ │ ⚙️ System Inputs (Tier 2)                              │ │
│ │ ├─ learning_rate: 3e-05                                 │ │
│ │ ├─ batch_size: 2                                        │ │
│ │ └─ max_epochs: 3                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### 2. **Dynamic Lists with Add/Remove**

```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Data Sources Specification                              │
│                                                             │
│ 📅 Time Range (Tier 1)                                    │
│ ├─ start_date: 2025-01-01T00:00:00                        │
│ └─ end_date: 2025-04-17T00:00:00                          │
│                                                             │
│ 📋 Data Sources (List[DataSourceConfig])                  │
│ ┌─ Item 1: MDS Data Source ──────────────────────── [×] ─┐ │
│ │ ├─ data_source_name: RAW_MDS_NA                        │ │
│ │ ├─ data_source_type: MDS                               │ │
│ │ └─ ▼ MDS Properties                                    │ │
│ │   ├─ service_name: AtoZ                                │ │
│ │   ├─ region: NA                                        │ │
│ │   └─ output_schema: [objectId, transactionDate, ...]  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─ Item 2: EDX Data Source ──────────────────────── [×] ─┐ │
│ │ ├─ data_source_name: TAGS                              │ │
│ │ ├─ data_source_type: EDX                               │ │
│ │ └─ ▼ EDX Properties                                    │ │
│ │   ├─ edx_provider: buyer-abuse                         │ │
│ │   ├─ edx_subject: tags                                 │ │
│ │   └─ edx_dataset: transaction-tags                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [+ Add Data Source ▼] [MDS] [EDX] [ANDES]                 │
└─────────────────────────────────────────────────────────────┘
```

#### 3. **Conditional Fields**

```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Data Source Configuration                               │
│                                                             │
│ data_source_type: [MDS ▼] [EDX] [ANDES]                   │
│                                                             │
│ ┌─ MDS Properties (shown because type=MDS) ───────────────┐ │
│ │ service_name: AtoZ                                      │ │
│ │ region: [NA ▼] [EU] [FE]                               │ │
│ │ org_id: 0                                               │ │
│ │ use_hourly_edx_data_set: ☐                             │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ┌─ EDX Properties (hidden because type≠EDX) ─────────────┐ │
│ │ [Hidden - would show EDX-specific fields]              │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1-2)
1. **NestedStructureAnalyzer**: Implement Pydantic model analysis
2. **HierarchicalConfigWidget**: Basic widget tree construction
3. **WidgetNode**: Tree node structure for nested widgets
4. **Basic Display**: Simple expandable sections for nested models

### Phase 2: Dynamic Lists (Week 3)
1. **DynamicListManager**: Add/remove functionality for List[BaseModel]
2. **List Item Templates**: Reusable templates for list items
3. **Validation**: List-level validation (min/max items, item validation)
4. **UI Polish**: Drag-and-drop reordering, better visual design

### Phase 3: Advanced Features (Week 4)
1. **ConditionalFieldManager**: Show/hide fields based on conditions
2. **Enhanced Inheritance**: Better inheritance-aware field categorization
3. **Performance**: Lazy loading, virtualization for large lists
4. **Error Handling**: Comprehensive error states and recovery

### Phase 4: Integration & Polish (Week 5)
1. **Existing Integration**: Update current config UI to use new system
2. **Migration Path**: Smooth transition from current flat approach
3. **Documentation**: Comprehensive usage examples
4. **Testing**: Unit tests, integration tests, user testing

## Key Benefits

### 1. **Consistent UX**
- All nested configs get the same hierarchical treatment
- Uniform patterns for nested models, lists, and conditional fields
- Predictable user experience across different config types

### 2. **Maintainable**
- Changes to nested structures automatically reflected in UI
- No manual field definitions required
- Single source of truth (Pydantic models)

### 3. **Scalable**
- Works for any depth of nesting
- Handles complex structures like List[Union[ModelA, ModelB]]
- Performance optimizations for large nested forms

### 4. **Type-Safe**
- Leverages Pydantic's type system for validation
- Full type information preserved through nesting levels
- IDE support and autocompletion

### 5. **Flexible**
- Multiple display modes (expandable, inline, modal, tabs)
- Customizable field categorization
- Extensible for new field types and patterns

## Migration Strategy

### Backward Compatibility
- Keep existing comprehensive field definitions for CradleDataLoadingConfig
- Gradually migrate other configs to use hierarchical system
- Provide fallback to flat display for unsupported structures

### Incremental Adoption
1. **Start with simple nested models** (PyTorchTrainingConfig + ModelHyperparameters)
2. **Add dynamic list support** for configs with List[BaseModel] fields
3. **Implement conditional fields** for Union types and Optional fields
4. **Full migration** once all patterns are supported

This design provides a robust, research-based foundation for handling any nested Pydantic configuration structure while maintaining the quality and usability of our current system.
