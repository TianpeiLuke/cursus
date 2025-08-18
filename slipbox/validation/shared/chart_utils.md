---
tags:
  - test
  - validation
  - utilities
  - visualization
  - charts
keywords:
  - chart utilities
  - validation visualization
  - alignment charts
  - scoring charts
  - matplotlib integration
topics:
  - validation utilities
  - data visualization
  - chart generation
  - scoring visualization
language: python
date of note: 2025-08-18
---

# Chart Utilities

The Chart Utilities module provides visualization capabilities for the Cursus validation framework, enabling the generation of alignment score charts and validation result visualizations. It integrates with matplotlib to create informative charts that help visualize validation performance and trends.

## Overview

The Chart Utilities module supports the validation framework by providing visual representations of validation results, alignment scores, and quality metrics. These visualizations help developers and stakeholders quickly understand validation status and identify areas for improvement.

## Key Features

### Alignment Score Charts

The module generates comprehensive alignment score charts that visualize:

- **Overall Alignment Scores**: Combined scores across all validation levels
- **Level-Specific Scores**: Individual scores for each alignment level
- **Quality Ratings**: Visual representation of quality ratings
- **Trend Analysis**: Score trends over time (when historical data available)

### Chart Types

#### Bar Charts
- **Level Comparison**: Compare scores across different alignment levels
- **Script Comparison**: Compare scores across different scripts
- **Quality Distribution**: Show distribution of quality ratings

#### Line Charts
- **Score Trends**: Track score changes over time
- **Performance Trends**: Monitor validation performance trends
- **Improvement Tracking**: Visualize improvement progress

#### Pie Charts
- **Quality Distribution**: Show proportion of different quality ratings
- **Issue Distribution**: Visualize distribution of issue types
- **Test Result Distribution**: Show pass/fail ratios

## Implementation

### Chart Generation Process

The chart utilities follow a standardized process for generating visualizations:

1. **Data Preparation**: Extract and format validation data
2. **Chart Configuration**: Set up chart parameters and styling
3. **Rendering**: Generate the chart using matplotlib
4. **Export**: Save chart to specified format and location
5. **Cleanup**: Clean up resources and temporary data

### Chart Configuration

```python
# Standard chart configuration
CHART_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'background_color': 'white',
    'title_font_size': 16,
    'label_font_size': 12,
    'legend_font_size': 10,
    'color_scheme': 'professional'
}
```

### Color Schemes

#### Professional Color Scheme
- **Excellent**: Deep Green (#2E7D32)
- **Good**: Light Green (#66BB6A)
- **Fair**: Orange (#FF9800)
- **Poor**: Red (#F44336)
- **Error**: Dark Red (#C62828)

#### Accessibility Color Scheme
- High contrast colors for better accessibility
- Colorblind-friendly palette
- Clear visual distinctions

## Chart Types and Usage

### Alignment Score Chart

The primary chart type for visualizing alignment validation results:

```python
def generate_alignment_score_chart(
    script_name: str,
    scores: Dict[str, float],
    output_dir: str,
    chart_title: Optional[str] = None
) -> str:
    """
    Generate alignment score chart for a specific script.
    
    Args:
        script_name: Name of the script being validated
        scores: Dictionary of level scores
        output_dir: Directory to save the chart
        chart_title: Optional custom chart title
        
    Returns:
        Path to the generated chart file
    """
```

**Chart Elements**:
- **Title**: Script name and overall score
- **Level Bars**: Individual bars for each alignment level
- **Score Labels**: Numeric scores displayed on bars
- **Quality Indicators**: Color coding based on quality ratings
- **Legend**: Explanation of colors and ratings

### Validation Summary Chart

Overview chart showing validation results across multiple scripts:

```python
def generate_validation_summary_chart(
    validation_results: Dict[str, Dict],
    output_dir: str
) -> str:
    """
    Generate summary chart for multiple validation results.
    
    Args:
        validation_results: Dictionary of validation results by script
        output_dir: Directory to save the chart
        
    Returns:
        Path to the generated chart file
    """
```

### Trend Analysis Chart

Historical trend chart for tracking validation improvements:

```python
def generate_trend_chart(
    historical_data: List[Dict],
    script_name: str,
    output_dir: str
) -> str:
    """
    Generate trend chart showing score changes over time.
    
    Args:
        historical_data: List of historical validation results
        script_name: Name of the script
        output_dir: Directory to save the chart
        
    Returns:
        Path to the generated chart file
    """
```

## Chart Styling

### Professional Styling

The charts use professional styling suitable for reports and presentations:

```python
# Chart styling configuration
PROFESSIONAL_STYLE = {
    'font_family': 'Arial',
    'title_weight': 'bold',
    'grid_alpha': 0.3,
    'bar_edge_color': 'black',
    'bar_edge_width': 0.5,
    'legend_frame': True,
    'tight_layout': True
}
```

### Color Coding

Quality-based color coding provides immediate visual feedback:

```python
# Quality color mapping
QUALITY_COLORS = {
    'Excellent': '#2E7D32',  # Deep Green
    'Good': '#66BB6A',       # Light Green
    'Fair': '#FF9800',       # Orange
    'Poor': '#F44336',       # Red
    'Error': '#C62828'       # Dark Red
}

# Score-based color selection
def get_score_color(score: float) -> str:
    if score >= 90:
        return QUALITY_COLORS['Excellent']
    elif score >= 75:
        return QUALITY_COLORS['Good']
    elif score >= 60:
        return QUALITY_COLORS['Fair']
    else:
        return QUALITY_COLORS['Poor']
```

## Integration with Validation Framework

### Alignment Report Integration

```python
class AlignmentReport:
    def export_with_chart(self, output_dir: str, script_name: str):
        # Generate standard report
        report_path = self.export_to_json(output_dir)
        
        # Generate accompanying chart
        chart_path = generate_alignment_score_chart(
            script_name=script_name,
            scores=self.get_level_scores(),
            output_dir=output_dir
        )
        
        return report_path, chart_path
```

### Universal Test Integration

```python
class UniversalStepBuilderTest:
    def export_results_with_chart(self, output_path: str):
        # Export test results
        results = self.run_all_tests_with_full_report()
        
        # Generate chart if scoring is available
        if 'scoring' in results:
            chart_path = generate_scoring_chart(
                results['scoring'],
                output_path
            )
            results['chart_path'] = chart_path
        
        return results
```

## Chart Output Formats

### Supported Formats

- **PNG**: High-quality raster images (default)
- **SVG**: Scalable vector graphics
- **PDF**: Print-ready format
- **JPG**: Compressed raster images

### File Naming Convention

```python
# Chart file naming pattern
chart_filename = f"{script_name}_alignment_score_chart.png"

# Examples
"tabular_preprocessing_alignment_score_chart.png"
"xgboost_training_alignment_score_chart.png"
"model_evaluation_alignment_score_chart.png"
```

### Directory Organization

```python
# Recommended directory structure for charts
reports/
├── charts/
│   ├── tabular_preprocessing_alignment_score_chart.png
│   ├── xgboost_training_alignment_score_chart.png
│   └── validation_summary_chart.png
├── json/
│   ├── tabular_preprocessing_alignment_report.json
│   └── xgboost_training_alignment_report.json
└── html/
    ├── tabular_preprocessing_alignment_report.html
    └── xgboost_training_alignment_report.html
```

## Usage Examples

### Basic Chart Generation

```python
from cursus.validation.shared.chart_utils import generate_alignment_score_chart

# Generate chart for validation results
scores = {
    'level1_script_contract': 95.0,
    'level2_contract_spec': 88.0,
    'level3_spec_dependencies': 92.0,
    'level4_builder_config': 85.0
}

chart_path = generate_alignment_score_chart(
    script_name='tabular_preprocessing',
    scores=scores,
    output_dir='reports/charts'
)

print(f"Chart saved to: {chart_path}")
```

### Batch Chart Generation

```python
# Generate charts for multiple scripts
validation_results = {
    'tabular_preprocessing': {...},
    'xgboost_training': {...},
    'model_evaluation': {...}
}

chart_paths = []
for script_name, results in validation_results.items():
    if 'scores' in results:
        chart_path = generate_alignment_score_chart(
            script_name=script_name,
            scores=results['scores'],
            output_dir='reports/charts'
        )
        chart_paths.append(chart_path)

print(f"Generated {len(chart_paths)} charts")
```

### Custom Chart Configuration

```python
# Custom chart with specific styling
chart_path = generate_alignment_score_chart(
    script_name='custom_script',
    scores=scores,
    output_dir='reports',
    chart_config={
        'figure_size': (10, 6),
        'color_scheme': 'accessibility',
        'title': 'Custom Validation Results',
        'show_grid': True,
        'export_format': 'svg'
    }
)
```

## Error Handling

### Matplotlib Availability

The chart utilities gracefully handle cases where matplotlib is not available:

```python
def generate_chart_safe(script_name: str, scores: Dict, output_dir: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        # Generate chart...
        return chart_path
    except ImportError:
        print("⚠️  Chart generation skipped (matplotlib not available)")
        return None
    except Exception as e:
        print(f"⚠️  Chart generation failed: {e}")
        return None
```

### File System Errors

```python
def ensure_output_directory(output_dir: str) -> bool:
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError:
        print(f"❌ Permission denied: Cannot create directory {output_dir}")
        return False
    except Exception as e:
        print(f"❌ Failed to create directory {output_dir}: {e}")
        return False
```

## Performance Considerations

### Memory Management

- **Figure Cleanup**: Properly close matplotlib figures to prevent memory leaks
- **Data Optimization**: Optimize data structures for chart generation
- **Batch Processing**: Efficiently generate multiple charts

### File I/O Optimization

- **Directory Caching**: Cache directory existence checks
- **Parallel Generation**: Generate multiple charts in parallel when possible
- **Compression**: Use appropriate compression for different formats

## Configuration Options

### Chart Appearance

```python
# Customize chart appearance
chart_config = {
    'figure_size': (12, 8),
    'dpi': 300,
    'title_font_size': 16,
    'label_font_size': 12,
    'color_scheme': 'professional',
    'show_grid': True,
    'grid_alpha': 0.3
}
```

### Export Options

```python
# Configure export settings
export_config = {
    'format': 'png',
    'quality': 95,
    'transparent': False,
    'bbox_inches': 'tight',
    'pad_inches': 0.1
}
```

## Best Practices

### Chart Design

1. **Clear Titles**: Use descriptive titles that explain the chart content
2. **Readable Labels**: Ensure all labels are clearly readable
3. **Appropriate Colors**: Use colors that convey meaning effectively
4. **Consistent Styling**: Maintain consistent styling across all charts
5. **Accessibility**: Consider colorblind users and accessibility requirements

### Performance

1. **Resource Cleanup**: Always clean up matplotlib resources
2. **Error Handling**: Handle missing dependencies gracefully
3. **File Management**: Organize chart files systematically
4. **Memory Usage**: Monitor memory usage for large datasets
5. **Parallel Processing**: Use parallel processing for batch operations

### Integration

1. **Consistent Naming**: Use consistent file naming conventions
2. **Directory Structure**: Organize charts in logical directory structures
3. **Metadata**: Include metadata in chart files when possible
4. **Version Control**: Consider version control implications for generated files
5. **Documentation**: Document chart generation processes clearly

## Future Enhancements

Planned improvements to the Chart Utilities:

1. **Interactive Charts**: Support for interactive web-based charts
2. **Advanced Analytics**: More sophisticated trend analysis and forecasting
3. **Custom Templates**: User-defined chart templates and themes
4. **Real-time Updates**: Live updating charts for continuous validation
5. **Export Formats**: Additional export formats and customization options

The Chart Utilities module enhances the validation framework by providing clear, informative visualizations that help developers and stakeholders understand validation results and track quality improvements over time.
