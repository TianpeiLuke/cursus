---
tags:
  - code
  - implementation
  - pipeline_catalog
  - pipeline_advisor
  - recommendations
  - intelligent_guidance
keywords:
  - Pipeline recommendations
  - Gap analysis
  - Use case matching
  - Upgrade paths
  - Intelligent advisor
topics:
  - Pipeline catalog advisor
  - Intelligent recommendations
  - Gap analysis
  - Use case matching
language: python
date of note: 2025-12-01
---

# Pipeline Advisor

## Overview

The `PipelineAdvisor` class provides intelligent pipeline recommendations based on user requirements, gap analysis, and use case matching. It guides users toward appropriate pipelines through analysis of needs and available options.

Key capabilities include requirements → recommendations matching, gap analysis between current and desired state, upgrade path suggestions, use case → pipeline mapping, feature requirement analysis, and complexity recommendations.

## Purpose and Major Tasks

### Primary Purpose
Intelligently recommend pipelines based on user requirements, provide gap analysis, and suggest upgrade paths to guide users toward appropriate pipeline choices.

### Major Tasks

1. **Requirements Analysis**: Parse and analyze user requirements
2. **Recommendation Generation**: Match requirements to pipelines
3. **Gap Analysis**: Identify missing features or capabilities
4. **Upgrade Path Suggestions**: Recommend evolution paths
5. **Use Case Matching**: Map use cases to pipelines
6. **Feature Requirements**: Analyze feature combinations
7. **Complexity Assessment**: Recommend appropriate complexity level
8. **Alternative Suggestions**: Provide backup options

## Module Contract

### Entry Point
```python
from cursus.pipeline_catalog.core.pipeline_advisor import PipelineAdvisor
```

### Class Initialization

```python
advisor = PipelineAdvisor(
    discovery: DAGAutoDiscovery,       # DAG discovery for catalog
    graph: PipelineKnowledgeGraph      # Knowledge graph for relationships
)
```

### Key Methods

```python
# Get recommendations from requirements
recommendations = advisor.recommend(
    requirements: Dict[str, Any],
    limit: int = 5
) -> List[Dict[str, Any]]

# Analyze gap between current and target
gap_analysis = advisor.analyze_gap(
    current_dag_id: str,
    target_requirements: Dict[str, Any]
) -> Dict[str, Any]

# Suggest upgrade path
upgrade_path = advisor.suggest_upgrade_path(
    current_dag_id: str,
    target_complexity: str
) -> Dict[str, Any]

# Match use case to pipelines
matches = advisor.match_use_case(
    use_case: str
) -> List[Dict[str, Any]]

# Get feature recommendations
features = advisor.recommend_features(
    current_features: List[str],
    goal: str
) -> List[str]
```

## Key Functions and Algorithms

### Requirement-Based Recommendations

#### `recommend(requirements, limit) -> List[Dict]`
**Purpose**: Generate pipeline recommendations based on requirements

**Algorithm**:
```python
1. Parse requirements:
   a. framework (required/preferred)
   b. features (must-have/nice-to-have)
   c. complexity (min/max)
   d. constraints (node count, etc.)
2. Score each pipeline:
   a. Framework match: +20 points (required) or +10 (preferred)
   b. Must-have features: +15 points each
   c. Nice-to-have features: +5 points each
   d. Complexity match: +10 points
   e. Constraint violations: -50 points
3. Sort by score (descending)
4. Return top N recommendations with explanations
```

**Requirements Format**:
```python
{
    "framework": {
        "required": "xgboost",  # or null
        "preferred": ["xgboost", "lightgbm"]
    },
    "features": {
        "must_have": ["training", "evaluation"],
        "nice_to_have": ["calibration"]
    },
    "complexity": {
        "min": "standard",
        "max": "comprehensive"
    },
    "constraints": {
        "max_nodes": 15,
        "max_edges": 20
    }
}
```

**Returns**:
```python
[
    {
        "dag_id": "xgboost_complete_e2e",
        "score": 65,
        "match_reasons": [
            "Framework match (required): xgboost",
            "Has all must-have features",
            "Complexity within range",
            "Satisfies all constraints"
        ],
        "missing_features": [],
        "framework": "xgboost",
        "complexity": "comprehensive"
    },
    ...
]
```

### Gap Analysis

#### `analyze_gap(current_dag_id, target_requirements) -> Dict`
**Purpose**: Analyze gaps between current pipeline and target requirements

**Algorithm**:
```python
1. Load current pipeline metadata
2. Compare with target requirements:
   a. Missing features
   b. Framework mismatch
   c. Complexity gap
   d. Constraint violations
3. Calculate gap severity:
   a. Critical: framework mismatch
   b. High: missing must-have features
   c. Medium: missing nice-to-have features
   d. Low: complexity difference
4. Generate recommendations to close gaps
5. Return gap analysis with action items
```

**Returns**:
```python
{
    "current": {
        "dag_id": "xgboost_simple",
        "framework": "xgboost",
        "complexity": "simple",
        "features": ["training"]
    },
    "target": {
        "framework": "xgboost",
        "features": ["training", "evaluation", "calibration"],
        "complexity": "comprehensive"
    },
    "gaps": [
        {
            "type": "missing_feature",
            "severity": "high",
            "item": "evaluation",
            "recommendation": "Upgrade to xgboost_training_with_evaluation"
        },
        {
            "type": "missing_feature",
            "severity": "high",
            "item": "calibration",
            "recommendation": "Further upgrade to xgboost_complete_e2e"
        },
        {
            "type": "complexity_gap",
            "severity": "medium",
            "current": "simple",
            "target": "comprehensive",
            "recommendation": "Follow evolution path"
        }
    ],
    "recommended_path": [
        "xgboost_simple",
        "xgboost_training_with_evaluation",
        "xgboost_complete_e2e"
    ]
}
```

### Upgrade Path Suggestions

#### `suggest_upgrade_path(current_dag_id, target_complexity) -> Dict`
**Purpose**: Suggest step-by-step upgrade path

**Algorithm**:
```python
1. Load current pipeline
2. Use knowledge graph to find evolution path
3. For each step in path:
   a. Identify new features added
   b. Estimate complexity increase
   c. List breaking changes
   d. Provide migration tips
4. Calculate total effort
5. Return detailed upgrade plan
```

**Returns**:
```python
{
    "current": "xgboost_simple",
    "target": "xgboost_complete_e2e",
    "path": [
        {
            "from": "xgboost_simple",
            "to": "xgboost_training_with_evaluation",
            "added_features": ["evaluation"],
            "effort": "low",
            "breaking_changes": [],
            "migration_tips": [
                "Add evaluation dataset",
                "Configure metrics"
            ]
        },
        {
            "from": "xgboost_training_with_evaluation",
            "to": "xgboost_complete_e2e",
            "added_features": ["calibration", "registration"],
            "effort": "medium",
            "breaking_changes": ["config schema change"],
            "migration_tips": [
                "Update config for calibration",
                "Set up model registry"
            ]
        }
    ],
    "total_effort": "medium",
    "estimated_time": "2-3 hours"
}
```

### Use Case Matching

#### `match_use_case(use_case) -> List[Dict]`
**Purpose**: Match textual use case description to pipelines

**Algorithm**:
```python
1. Tokenize use case description
2. Extract key concepts:
   a. Problem type (classification, regression, etc.)
   b. Framework mentions
   c. Required capabilities
   d. Scale indicators
3. Score pipelines against concepts
4. Rank by relevance
5. Return matches with explanations
```

**Example Use Cases**:
```python
# Classification with evaluation
"I need to train a binary classifier and evaluate its performance on a test set"

# Incremental improvement
"I have a simple training pipeline but need to add model calibration"

# Production deployment
"Need a comprehensive pipeline for production deployment with monitoring"
```

**Returns**:
```python
[
    {
        "dag_id": "xgboost_training_with_evaluation",
        "relevance_score": 85,
        "match_reasons": [
            "Supports binary classification",
            "Includes evaluation step",
            "Test set evaluation capability"
        ],
        "use_case_fit": "excellent",
        "framework": "xgboost"
    },
    ...
]
```

### Feature Recommendations

#### `recommend_features(current_features, goal) -> List[str]`
**Purpose**: Recommend additional features based on goal

**Algorithm**:
```python
1. Analyze current features
2. Parse goal description
3. Identify feature gaps for goal
4. Rank recommendations by:
   a. Goal relevance
   b. Common patterns
   c. Dependencies
5. Return ordered feature list
```

**Goals**:
- "production_ready": Add calibration, monitoring, registration
- "better_evaluation": Add metrics, visualization, comparison
- "robust_training": Add validation, early stopping, checkpoints

**Example**:
```python
current = ["training"]
goal = "production_ready"

recommendations = advisor.recommend_features(current, goal)
# Returns: ["evaluation", "calibration", "registration", "monitoring"]
```

## Integration Patterns

### With PipelineFactory

```python
from cursus.pipeline_catalog.core import PipelineAdvisor, PipelineFactory

advisor = PipelineAdvisor(discovery, graph)
factory = PipelineFactory()

# Get recommendations
requirements = {
    "framework": {"required": "xgboost"},
    "features": {"must_have": ["training", "evaluation"]}
}

recommendations = advisor.recommend(requirements, limit=3)

# Create recommended pipeline
best_match = recommendations[0]
pipeline = factory.create(best_match['dag_id'])
```

### Guided Pipeline Selection

```python
# Step 1: Get user requirements
requirements = get_user_requirements()  # Interactive prompts

# Step 2: Get recommendations
recommendations = advisor.recommend(requirements)

# Step 3: Show options
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['dag_id']} (score: {rec['score']})")
    for reason in rec['match_reasons']:
        print(f"   - {reason}")

# Step 4: User selects
chosen = recommendations[user_choice - 1]

# Step 5: Create pipeline
pipeline = factory.create(chosen['dag_id'])
```

### Upgrade Planning

```python
# Current pipeline needs enhancement
current = "xgboost_simple"
target_requirements = {
    "features": {"must_have": ["training", "evaluation", "calibration"]}
}

# Analyze gap
gap = advisor.analyze_gap(current, target_requirements)

print(f"Current: {gap['current']['dag_id']}")
print(f"Gaps identified: {len(gap['gaps'])}")

for gap_item in gap['gaps']:
    print(f"  - {gap_item['type']}: {gap_item['item']}")
    print(f"    Recommendation: {gap_item['recommendation']}")

# Suggest upgrade path
upgrade = advisor.suggest_upgrade_path(current, "comprehensive")

print(f"\nUpgrade path ({len(upgrade['path'])} steps):")
for step in upgrade['path']:
    print(f"  {step['from']} → {step['to']}")
    print(f"    Adds: {step['added_features']}")
    print(f"    Effort: {step['effort']}")
```

## Best Practices

### 1. Start with Clear Requirements

```python
# ✅ Good: Specific requirements
requirements = {
    "framework": {"required": "xgboost"},
    "features": {
        "must_have": ["training", "evaluation"],
        "nice_to_have": ["calibration"]
    },
    "complexity": {"min": "standard"}
}

recommendations = advisor.recommend(requirements)
```

### 2. Analyze Before Upgrading

```python
# ✅ Good: Understand gaps first
gap = advisor.analyze_gap(
    current_dag_id="xgboost_simple",
    target_requirements=requirements
)

if gap['gaps']:
    print("Gaps to address:")
    for g in gap['gaps']:
        print(f"  - {g['type']}: {g['item']}")
    
    # Then upgrade
    upgrade = advisor.suggest_upgrade_path(current, "comprehensive")
```

### 3. Use Case-Based Discovery

```python
# ✅ Good: Natural language use case
use_case = """
I need to train an XGBoost model on imbalanced data,
evaluate it on a test set, and calibrate the probabilities
for production deployment.
"""

matches = advisor.match_use_case(use_case)

# Review matches
for match in matches:
    print(f"{match['dag_id']}: {match['use_case_fit']}")
```

## Performance Characteristics

| Operation | Time Complexity | Typical Time |
|-----------|----------------|--------------|
| recommend | O(n * f) | ~10ms |
| analyze_gap | O(f) | ~5ms |
| suggest_upgrade_path | O(d) | ~5ms |
| match_use_case | O(n * t) | ~15ms |
| recommend_features | O(f²) | ~3ms |

Where:
- n = number of pipelines
- f = number of features
- d = path depth
- t = number of tokens

## Examples

### Example 1: Requirement-Based Selection

```python
from cursus.pipeline_catalog.core import PipelineAdvisor

advisor = PipelineAdvisor(discovery, graph)

# Define requirements
requirements = {
    "framework": {"preferred": ["xgboost", "lightgbm"]},
    "features": {
        "must_have": ["training", "evaluation"],
        "nice_to_have": ["calibration"]
    },
    "complexity": {"min": "standard", "max": "comprehensive"}
}

# Get recommendations
recommendations = advisor.recommend(requirements, limit=5)

print(f"Top {len(recommendations)} recommendations:")
for rec in recommendations:
    print(f"\n{rec['dag_id']} (score: {rec['score']})")
    print("Reasons:")
    for reason in rec['match_reasons']:
        print(f"  ✓ {reason}")
    if rec['missing_features']:
        print("Missing (nice-to-have):")
        for feat in rec['missing_features']:
            print(f"  ✗ {feat}")
```

### Example 2: Gap Analysis and Upgrade Planning

```python
# Analyze current pipeline
current = "xgboost_simple"
target_requirements = {
    "features": {
        "must_have": ["training", "evaluation", "calibration"]
    }
}

gap = advisor.analyze_gap(current, target_requirements)

print(f"Gap Analysis for {current}:")
print(f"Missing features: {len([g for g in gap['gaps'] if g['type'] == 'missing_feature'])}")

# Get upgrade path
if gap['recommended_path']:
    print(f"\nRecommended upgrade path:")
    for i, step in enumerate(gap['recommended_path']):
        print(f"  {i+1}. {step}")
    
    # Get detailed upgrade plan
    target = gap['recommended_path'][-1]
    upgrade = advisor.suggest_upgrade_path(current, "comprehensive")
    
    print(f"\nTotal effort: {upgrade['total_effort']}")
    print(f"Estimated time: {upgrade['estimated_time']}")
```

### Example 3: Use Case Matching

```python
# Natural language use case
use_case = """
Need to build a production ML pipeline for binary classification.
Must include training with cross-validation, comprehensive evaluation
with multiple metrics, and model calibration for probability accuracy.
"""

matches = advisor.match_use_case(use_case)

print(f"Found {len(matches)} matching pipelines:")
for match in matches[:3]:
    print(f"\n{match['dag_id']}")
    print(f"  Relevance: {match['relevance_score']}/100")
    print(f"  Fit: {match['use_case_fit']}")
    print("  Reasons:")
    for reason in match['match_reasons']:
        print(f"    • {reason}")
```

## References

### Related Components

- **[DAG Discovery](dag_discovery.md)**: Provides pipeline catalog
- **[Pipeline Factory](pipeline_factory.md)**: Creates recommended pipelines
- **[Pipeline Knowledge Graph](pipeline_knowledge_graph.md)**: Provides relationship data
- **[Pipeline Explorer](pipeline_explorer.md)**: Interactive exploration

### Design Documents

- **[Pipeline Catalog Redesign](../../1_design/pipeline_catalog_redesign.md)**: Overall system design
- **[Recommendation Engine Design](../../1_design/recommendation_engine.md)**: Recommendation algorithms
