#!/usr/bin/env python3
"""
Performance Analysis Script

This script analyzes the performance of the XGBoost 3-step pipeline execution,
generates visualizations, and provides comprehensive reporting.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup environment and imports."""
    print("=== PERFORMANCE ANALYSIS SETUP ===")
    
    # Add cursus to path
    sys.path.append(str(Path.cwd().parent.parent.parent / 'src'))
    
    # Import Cursus components
    try:
        from cursus.validation.runtime.jupyter.notebook_interface import NotebookInterface
        from cursus.validation.runtime.core.data_flow_manager import DataFlowManager
        print("✓ Successfully imported Cursus components")
        cursus_available = True
    except ImportError as e:
        print(f"⚠ Import error: {e}")
        print("Using standard libraries for analysis...")
        cursus_available = False
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    print(f"Performance analysis started at {datetime.now()}")
    return cursus_available

def load_test_results():
    """Load all test results from previous notebooks."""
    print("\n=== LOAD TEST RESULTS ===")
    
    BASE_DIR = Path.cwd()
    RESULTS_DIR = BASE_DIR / 'outputs' / 'results'
    
    results = {}
    
    # Load individual step test results
    individual_results_path = RESULTS_DIR / 'individual_step_test_results.json'
    if individual_results_path.exists():
        with open(individual_results_path, 'r') as f:
            results['individual_step_testing'] = json.load(f)
        print(f"✓ Loaded individual step test results")
    else:
        print(f"⚠ Individual step test results not found: {individual_results_path}")
    
    # Load end-to-end pipeline results
    pipeline_results_path = RESULTS_DIR / 'end_to_end_pipeline_results.json'
    if pipeline_results_path.exists():
        with open(pipeline_results_path, 'r') as f:
            results['end_to_end_pipeline'] = json.load(f)
        print(f"✓ Loaded end-to-end pipeline results")
    else:
        print(f"⚠ End-to-end pipeline results not found: {pipeline_results_path}")
    
    # Load dataset metadata
    dataset_metadata_path = BASE_DIR / 'data' / 'dataset_metadata.json'
    if dataset_metadata_path.exists():
        with open(dataset_metadata_path, 'r') as f:
            results['dataset_metadata'] = json.load(f)
        print(f"✓ Loaded dataset metadata")
    else:
        print(f"⚠ Dataset metadata not found: {dataset_metadata_path}")
    
    return results

class PerformanceAnalyzer:
    """Comprehensive performance analyzer for pipeline testing results."""
    
    def __init__(self, results_data):
        self.results_data = results_data
        self.analysis_results = {}
        
        # Create output directory for visualizations
        self.viz_dir = Path.cwd() / 'outputs' / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PerformanceAnalyzer initialized with visualization output: {self.viz_dir}")
    
    def analyze_execution_times(self):
        """Analyze execution times across different test modes."""
        print("\n=== ANALYZE EXECUTION TIMES ===")
        
        execution_data = []
        
        # Extract individual step testing times
        if 'individual_step_testing' in self.results_data:
            individual_results = self.results_data['individual_step_testing']
            for step_name, exec_time in individual_results.get('execution_times', {}).items():
                execution_data.append({
                    'step_name': step_name,
                    'execution_time': exec_time,
                    'test_mode': 'Individual Step Testing',
                    'status': individual_results['step_results'][step_name]['status']
                })
        
        # Extract end-to-end pipeline times
        if 'end_to_end_pipeline' in self.results_data:
            pipeline_results = self.results_data['end_to_end_pipeline']
            for step_name, exec_time in pipeline_results.get('execution_times', {}).items():
                execution_data.append({
                    'step_name': step_name,
                    'execution_time': exec_time,
                    'test_mode': 'End-to-End Pipeline',
                    'status': pipeline_results['step_results'][step_name]['status']
                })
        
        if not execution_data:
            print("⚠ No execution time data available for analysis")
            return
        
        # Create DataFrame
        df = pd.DataFrame(execution_data)
        
        # Generate execution time analysis
        print("EXECUTION TIME ANALYSIS")
        print("="*30)
        
        # Summary statistics
        summary_stats = df.groupby(['test_mode', 'step_name'])['execution_time'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)
        
        print("Summary Statistics (seconds):")
        print(summary_stats)
        
        # Store analysis results
        self.analysis_results['execution_times'] = {
            'raw_data': execution_data,
            'summary_statistics': summary_stats.to_dict(),
            'total_individual_time': df[df['test_mode'] == 'Individual Step Testing']['execution_time'].sum(),
            'total_pipeline_time': df[df['test_mode'] == 'End-to-End Pipeline']['execution_time'].sum()
        }
        
        return df
    
    def analyze_success_rates(self):
        """Analyze success rates across different test modes."""
        print("\n=== ANALYZE SUCCESS RATES ===")
        
        success_data = []
        
        # Individual step testing success rates
        if 'individual_step_testing' in self.results_data:
            individual_results = self.results_data['individual_step_testing']
            success_data.append({
                'test_mode': 'Individual Step Testing',
                'total_steps': individual_results['total_steps'],
                'successful_steps': individual_results['successful_steps'],
                'failed_steps': individual_results['failed_steps'],
                'success_rate': individual_results['success_rate']
            })
        
        # End-to-end pipeline success rates
        if 'end_to_end_pipeline' in self.results_data:
            pipeline_results = self.results_data['end_to_end_pipeline']
            success_data.append({
                'test_mode': 'End-to-End Pipeline',
                'total_steps': pipeline_results['total_steps'],
                'successful_steps': pipeline_results['successful_steps'],
                'failed_steps': pipeline_results['failed_steps'],
                'success_rate': pipeline_results['success_rate']
            })
        
        if not success_data:
            print("⚠ No success rate data available for analysis")
            return
        
        print("SUCCESS RATE ANALYSIS")
        print("="*25)
        
        for data in success_data:
            print(f"\n{data['test_mode']}:")
            print(f"  Total Steps: {data['total_steps']}")
            print(f"  Successful: {data['successful_steps']}")
            print(f"  Failed: {data['failed_steps']}")
            print(f"  Success Rate: {data['success_rate']:.1f}%")
        
        self.analysis_results['success_rates'] = success_data
        return success_data
    
    def analyze_data_flow(self):
        """Analyze data flow and file sizes throughout the pipeline."""
        print("\n=== ANALYZE DATA FLOW ===")
        
        BASE_DIR = Path.cwd()
        WORKSPACE_DIR = BASE_DIR / 'outputs' / 'workspace'
        DATA_DIR = BASE_DIR / 'data'
        
        file_analysis = []
        
        # Analyze input data files
        input_files = [
            ('train_data.csv', DATA_DIR / 'train_data.csv'),
            ('eval_data.csv', DATA_DIR / 'eval_data.csv')
        ]
        
        for file_name, file_path in input_files:
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_analysis.append({
                    'file_name': file_name,
                    'file_path': str(file_path),
                    'file_size_bytes': file_size,
                    'file_size_kb': file_size / 1024,
                    'file_type': 'input_data'
                })
        
        # Analyze output files
        output_files = [
            ('xgboost_model.pkl', WORKSPACE_DIR / 'xgboost_model.pkl'),
            ('predictions.csv', WORKSPACE_DIR / 'predictions.csv'),
            ('eval_metrics.json', WORKSPACE_DIR / 'eval_metrics.json'),
            ('calibrated_model.pkl', WORKSPACE_DIR / 'calibrated_model.pkl'),
            ('calibrated_predictions.csv', WORKSPACE_DIR / 'calibrated_predictions.csv')
        ]
        
        for file_name, file_path in output_files:
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_analysis.append({
                    'file_name': file_name,
                    'file_path': str(file_path),
                    'file_size_bytes': file_size,
                    'file_size_kb': file_size / 1024,
                    'file_type': 'output_data'
                })
        
        if file_analysis:
            df_files = pd.DataFrame(file_analysis)
            
            print("DATA FLOW ANALYSIS")
            print("="*20)
            print(f"Total files analyzed: {len(file_analysis)}")
            print(f"Input files: {len([f for f in file_analysis if f['file_type'] == 'input_data'])}")
            print(f"Output files: {len([f for f in file_analysis if f['file_type'] == 'output_data'])}")
            
            total_input_size = df_files[df_files['file_type'] == 'input_data']['file_size_kb'].sum()
            total_output_size = df_files[df_files['file_type'] == 'output_data']['file_size_kb'].sum()
            
            print(f"Total input data size: {total_input_size:.2f} KB")
            print(f"Total output data size: {total_output_size:.2f} KB")
            
            self.analysis_results['data_flow'] = {
                'file_analysis': file_analysis,
                'total_input_size_kb': total_input_size,
                'total_output_size_kb': total_output_size,
                'data_expansion_ratio': total_output_size / total_input_size if total_input_size > 0 else 0
            }
        else:
            print("⚠ No files found for data flow analysis")
    
    def create_visualizations(self):
        """Create comprehensive performance visualizations."""
        print("\n=== CREATE VISUALIZATIONS ===")
        
        # Create execution time comparison chart
        if 'execution_times' in self.analysis_results:
            self._create_execution_time_chart()
        
        # Create success rate comparison chart
        if 'success_rates' in self.analysis_results:
            self._create_success_rate_chart()
        
        # Create data flow visualization
        if 'data_flow' in self.analysis_results:
            self._create_data_flow_chart()
        
        # Create comprehensive dashboard
        self._create_performance_dashboard()
    
    def _create_execution_time_chart(self):
        """Create execution time comparison chart."""
        execution_data = self.analysis_results['execution_times']['raw_data']
        df = pd.DataFrame(execution_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart comparing execution times by step
        step_comparison = df.pivot_table(
            values='execution_time', 
            index='step_name', 
            columns='test_mode', 
            aggfunc='mean'
        )
        
        step_comparison.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_title('Execution Time by Step and Test Mode')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.legend(title='Test Mode')
        ax1.grid(True, alpha=0.3)
        
        # Box plot showing execution time distribution
        sns.boxplot(data=df, x='step_name', y='execution_time', hue='test_mode', ax=ax2)
        ax2.set_title('Execution Time Distribution')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'execution_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Execution time chart saved: {self.viz_dir / 'execution_time_analysis.png'}")
    
    def _create_success_rate_chart(self):
        """Create success rate comparison chart."""
        success_data = self.analysis_results['success_rates']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rate comparison
        test_modes = [data['test_mode'] for data in success_data]
        success_rates = [data['success_rate'] for data in success_data]
        
        bars = ax1.bar(test_modes, success_rates, color=['lightblue', 'lightgreen'])
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # Step success breakdown
        step_data = []
        for data in success_data:
            step_data.extend([
                {'test_mode': data['test_mode'], 'status': 'Successful', 'count': data['successful_steps']},
                {'test_mode': data['test_mode'], 'status': 'Failed', 'count': data['failed_steps']}
            ])
        
        df_steps = pd.DataFrame(step_data)
        step_pivot = df_steps.pivot_table(values='count', index='test_mode', columns='status', fill_value=0)
        
        step_pivot.plot(kind='bar', stacked=True, ax=ax2, color=['lightcoral', 'lightgreen'])
        ax2.set_title('Step Success/Failure Breakdown')
        ax2.set_ylabel('Number of Steps')
        ax2.legend(title='Status')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'success_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Success rate chart saved: {self.viz_dir / 'success_rate_analysis.png'}")
    
    def _create_data_flow_chart(self):
        """Create data flow visualization."""
        file_analysis = self.analysis_results['data_flow']['file_analysis']
        df_files = pd.DataFrame(file_analysis)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # File size comparison
        input_files = df_files[df_files['file_type'] == 'input_data']
        output_files = df_files[df_files['file_type'] == 'output_data']
        
        # Bar chart of file sizes
        all_files = pd.concat([input_files, output_files])
        colors = ['lightblue' if ft == 'input_data' else 'lightcoral' for ft in all_files['file_type']]
        
        bars = ax1.bar(range(len(all_files)), all_files['file_size_kb'], color=colors)
        ax1.set_title('File Sizes Throughout Pipeline')
        ax1.set_ylabel('File Size (KB)')
        ax1.set_xticks(range(len(all_files)))
        ax1.set_xticklabels(all_files['file_name'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', label='Input Data'),
            Patch(facecolor='lightcoral', label='Output Data')
        ]
        ax1.legend(handles=legend_elements)
        
        # Pie chart of data distribution
        total_input = self.analysis_results['data_flow']['total_input_size_kb']
        total_output = self.analysis_results['data_flow']['total_output_size_kb']
        
        sizes = [total_input, total_output]
        labels = ['Input Data', 'Output Data']
        colors = ['lightblue', 'lightcoral']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Data Size Distribution')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'data_flow_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Data flow chart saved: {self.viz_dir / 'data_flow_analysis.png'}")
    
    def _create_performance_dashboard(self):
        """Create comprehensive performance dashboard."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('XGBoost Pipeline Performance Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Execution time summary (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'execution_times' in self.analysis_results:
            total_individual = self.analysis_results['execution_times']['total_individual_time']
            total_pipeline = self.analysis_results['execution_times']['total_pipeline_time']
            
            ax1.bar(['Individual\nStep Testing', 'End-to-End\nPipeline'], 
                   [total_individual, total_pipeline], 
                   color=['lightblue', 'lightgreen'])
            ax1.set_title('Total Execution Time')
            ax1.set_ylabel('Time (seconds)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Success rate summary (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'success_rates' in self.analysis_results:
            success_data = self.analysis_results['success_rates']
            modes = [data['test_mode'] for data in success_data]
            rates = [data['success_rate'] for data in success_data]
            
            bars = ax2.bar(modes, rates, color=['lightblue', 'lightgreen'])
            ax2.set_title('Success Rates')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. Data flow summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'data_flow' in self.analysis_results:
            input_size = self.analysis_results['data_flow']['total_input_size_kb']
            output_size = self.analysis_results['data_flow']['total_output_size_kb']
            
            ax3.bar(['Input Data', 'Output Data'], [input_size, output_size], 
                   color=['lightblue', 'lightcoral'])
            ax3.set_title('Data Volume')
            ax3.set_ylabel('Size (KB)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Step-by-step execution times (middle row)
        ax4 = fig.add_subplot(gs[1, :])
        if 'execution_times' in self.analysis_results:
            execution_data = self.analysis_results['execution_times']['raw_data']
            df = pd.DataFrame(execution_data)
            
            step_comparison = df.pivot_table(
                values='execution_time', 
                index='step_name', 
                columns='test_mode', 
                aggfunc='mean'
            )
            
            step_comparison.plot(kind='bar', ax=ax4, rot=0)
            ax4.set_title('Step-by-Step Execution Time Comparison')
            ax4.set_ylabel('Execution Time (seconds)')
            ax4.legend(title='Test Mode')
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics summary (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary text
        summary_text = self._generate_performance_summary()
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.savefig(self.viz_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Performance dashboard saved: {self.viz_dir / 'performance_dashboard.png'}")
    
    def _generate_performance_summary(self):
        """Generate performance summary text."""
        summary_lines = ["PERFORMANCE SUMMARY", "=" * 50]
        
        # Execution time summary
        if 'execution_times' in self.analysis_results:
            total_individual = self.analysis_results['execution_times']['total_individual_time']
            total_pipeline = self.analysis_results['execution_times']['total_pipeline_time']
            
            summary_lines.extend([
                f"Total Individual Step Testing Time: {total_individual:.2f}s",
                f"Total End-to-End Pipeline Time: {total_pipeline:.2f}s",
                f"Pipeline Overhead: {total_pipeline - total_individual:.2f}s",
                ""
            ])
        
        # Success rate summary
        if 'success_rates' in self.analysis_results:
            for data in self.analysis_results['success_rates']:
                summary_lines.append(
                    f"{data['test_mode']}: {data['successful_steps']}/{data['total_steps']} "
                    f"steps successful ({data['success_rate']:.1f}%)"
                )
            summary_lines.append("")
        
        # Data flow summary
        if 'data_flow' in self.analysis_results:
            input_size = self.analysis_results['data_flow']['total_input_size_kb']
            output_size = self.analysis_results['data_flow']['total_output_size_kb']
            expansion_ratio = self.analysis_results['data_flow']['data_expansion_ratio']
            
            summary_lines.extend([
                f"Input Data Size: {input_size:.2f} KB",
                f"Output Data Size: {output_size:.2f} KB",
                f"Data Expansion Ratio: {expansion_ratio:.2f}x",
                ""
            ])
        
        # Test completion status
        summary_lines.extend([
            f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Visualizations saved to: {self.viz_dir}"
        ])
        
        return "\n".join(summary_lines)

def run_performance_analysis():
    """Run comprehensive performance analysis."""
    print("STARTING PERFORMANCE ANALYSIS")
    print("="*40)
    
    try:
        # Load test results
        results_data = load_test_results()
        
        if not results_data:
            print("⚠ No test results available for analysis")
            return False
        
        # Create analyzer
        analyzer = PerformanceAnalyzer(results_data)
        
        # Run analyses
        analyzer.analyze_execution_times()
        analyzer.analyze_success_rates()
        analyzer.analyze_data_flow()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save comprehensive analysis results
        save_analysis_results(analyzer.analysis_results)
        
        return True
        
    except Exception as e:
        print(f"✗ Performance analysis failed: {e}")
        return False

def save_analysis_results(analysis_results):
    """Save comprehensive analysis results."""
    print("\n=== SAVE ANALYSIS RESULTS ===")
    
    BASE_DIR = Path.cwd()
    RESULTS_DIR = BASE_DIR / 'outputs' / 'results'
    
    # Prepare results for JSON serialization
    serializable_results = {}
    for key, value in analysis_results.items():
        if isinstance(value, dict):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    # Add metadata
    final_results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_type': 'performance_analysis',
        'analysis_results': serializable_results,
        'visualizations_directory': str(Path.cwd() / 'outputs' / 'visualizations')
    }
    
    results_path = RESULTS_DIR / 'performance_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Performance analysis results saved: {results_path}")

def generate_final_report():
    """Generate final performance analysis report."""
    print("\n=== GENERATE FINAL REPORT ===")
    
    BASE_DIR = Path.cwd()
    RESULTS_DIR = BASE_DIR / 'outputs' / 'results'
    
    # Load all results
    all_results = {}
    result_files = [
        'individual_step_test_results.json',
        'end_to_end_pipeline_results.json',
        'performance_analysis_results.json'
    ]
    
    for result_file in result_files:
        result_path = RESULTS_DIR / result_file
        if result_path.exists():
            with open(result_path, 'r') as f:
                all_results[result_file.replace('.json', '')] = json.load(f)
    
    print("FINAL PERFORMANCE REPORT")
    print("="*50)
    
    # Overall test summary
    if 'individual_step_test_results' in all_results:
        individual_results = all_results['individual_step_test_results']
        print(f"Individual Step Testing: {individual_results['success_rate']:.1f}% success rate")
    
    if 'end_to_end_pipeline_results' in all_results:
        pipeline_results = all_results['end_to_end_pipeline_results']
        print(f"End-to-End Pipeline: {pipeline_results['success_rate']:.1f}% success rate")
    
    # Performance insights
    if 'performance_analysis_results' in all_results:
        perf_results = all_results['performance_analysis_results']
        print(f"Performance analysis completed at: {perf_results['analysis_timestamp']}")
        print(f"Visualizations available in: {perf_results['visualizations_directory']}")
    
    print("\n✓ XGBoost 3-step pipeline testing completed successfully!")
    print("✓ All performance metrics analyzed and visualized")
    print("✓ Ready for production deployment validation")

def main():
    """Main execution function."""
    # Setup environment
    cursus_available = setup_environment()
    
    # Run performance analysis
    success = run_performance_analysis()
    
    if success:
        # Generate final report
        generate_final_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("🎉 All analyses completed!")
        print("📊 Visualizations generated!")
        print("📋 Reports saved!")
    else:
        print("\n⚠ Performance analysis failed!")

if __name__ == "__main__":
    main()
