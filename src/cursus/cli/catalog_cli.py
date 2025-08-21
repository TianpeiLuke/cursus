"""
Command-line interface for the Zettelkasten-based pipeline catalog.

This module provides CLI commands for working with the pipeline catalog using
Zettelkasten knowledge management principles, including connection-based navigation,
tag-based discovery, and intelligent recommendations.

Example usage:
    # Registry management
    cursus catalog registry validate
    cursus catalog registry stats
    cursus catalog registry export --pipelines xgb_training_simple,pytorch_training_basic
    
    # Discovery commands
    cursus catalog find --tags training,xgboost
    cursus catalog find --framework pytorch --complexity standard
    cursus catalog find --use-case "Basic PyTorch training workflow"
    
    # Connection navigation
    cursus catalog connections --pipeline xgb_training_simple
    cursus catalog alternatives --pipeline xgb_training_simple
    cursus catalog path --from xgb_training_simple --to xgb_e2e_comprehensive
    
    # Recommendations
    cursus catalog recommend --use-case "XGBoost training"
    cursus catalog recommend --next-steps xgb_training_simple
    cursus catalog recommend --learning-path --framework xgboost
"""

import os
import json
import argparse
import textwrap
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from ..pipeline_catalog.utils import (
    CatalogRegistry,
    ConnectionTraverser,
    TagBasedDiscovery,
    PipelineRecommendationEngine,
    RegistryValidator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def setup_parser(parser):
    """
    Set up the argument parser for the Zettelkasten catalog CLI.
    
    Args:
        parser: The argparse parser to configure
    """
    subparsers = parser.add_subparsers(dest='command', help='Catalog command')
    
    # Registry management commands
    registry_parser = subparsers.add_parser('registry', help='Registry management commands')
    registry_subparsers = registry_parser.add_subparsers(dest='registry_command', help='Registry command')
    
    # Registry validate command
    validate_parser = registry_subparsers.add_parser('validate', help='Validate registry integrity')
    validate_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                help='Output format')
    
    # Registry stats command
    stats_parser = registry_subparsers.add_parser('stats', help='Show registry statistics')
    stats_parser.add_argument('--format', choices=['text', 'json'], default='text',
                             help='Output format')
    
    # Registry export command
    export_parser = registry_subparsers.add_parser('export', help='Export pipeline metadata')
    export_parser.add_argument('--pipelines', help='Comma-separated list of pipeline IDs to export')
    export_parser.add_argument('--output', '-o', help='Output file path (default: stdout)')
    export_parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                              help='Export format')
    
    # Discovery commands
    find_parser = subparsers.add_parser('find', help='Find pipelines by criteria')
    find_parser.add_argument('--tags', help='Comma-separated list of tags to search for')
    find_parser.add_argument('--framework', help='Filter by framework (e.g., xgboost, pytorch)')
    find_parser.add_argument('--complexity', choices=['simple', 'standard', 'comprehensive'], 
                            help='Filter by complexity level')
    find_parser.add_argument('--use-case', help='Search by use case description')
    find_parser.add_argument('--features', help='Comma-separated list of features to search for')
    find_parser.add_argument('--mods-compatible', action='store_true',
                            help='Only show MODS-compatible pipelines')
    find_parser.add_argument('--format', choices=['table', 'json'], default='table',
                           help='Output format')
    find_parser.add_argument('--limit', type=int, default=20,
                           help='Maximum number of results to show')
    
    # Connection navigation commands
    connections_parser = subparsers.add_parser('connections', help='Show pipeline connections')
    connections_parser.add_argument('--pipeline', required=True, help='Pipeline ID to show connections for')
    connections_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                   help='Output format')
    
    alternatives_parser = subparsers.add_parser('alternatives', help='Show alternative pipelines')
    alternatives_parser.add_argument('--pipeline', required=True, help='Pipeline ID to find alternatives for')
    alternatives_parser.add_argument('--format', choices=['table', 'json'], default='table',
                                    help='Output format')
    
    path_parser = subparsers.add_parser('path', help='Find connection path between pipelines')
    path_parser.add_argument('--from', dest='source', required=True, help='Source pipeline ID')
    path_parser.add_argument('--to', dest='target', required=True, help='Target pipeline ID')
    path_parser.add_argument('--format', choices=['text', 'json'], default='text',
                           help='Output format')
    
    # Recommendation commands
    recommend_parser = subparsers.add_parser('recommend', help='Get pipeline recommendations')
    recommend_parser.add_argument('--use-case', help='Get recommendations for a specific use case')
    recommend_parser.add_argument('--next-steps', help='Get next step recommendations from a pipeline ID')
    recommend_parser.add_argument('--learning-path', action='store_true',
                                 help='Generate a learning path')
    recommend_parser.add_argument('--framework', help='Framework for learning path (required with --learning-path)')
    recommend_parser.add_argument('--skill-level', choices=['beginner', 'intermediate', 'advanced'],
                                 help='Target skill level for recommendations')
    recommend_parser.add_argument('--format', choices=['text', 'json'], default='text',
                                 help='Output format')
    recommend_parser.add_argument('--limit', type=int, default=5,
                                 help='Maximum number of recommendations')
    
    # Legacy commands for backward compatibility
    list_parser = subparsers.add_parser('list', help='List all available pipelines')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table',
                           help='Output format')
    list_parser.add_argument('--sort', choices=['id', 'title', 'framework', 'complexity'], 
                            default='id', help='Sort field')
    
    show_parser = subparsers.add_parser('show', help='Show details for a pipeline')
    show_parser.add_argument('pipeline_id', help='ID of the pipeline to show')
    show_parser.add_argument('--format', choices=['text', 'json'], default='text',
                           help='Output format')
    
    return parser


# ============================================================================
# Registry Management Commands
# ============================================================================

def registry_validate(args):
    """Validate registry integrity."""
    try:
        validator = RegistryValidator()
        validation_result = validator.validate_registry()
        
        if args.format == 'json':
            print(json.dumps(validation_result.to_dict(), indent=2))
        else:
            print("\nRegistry Validation Results:")
            print("=" * 30)
            
            if validation_result.is_valid:
                print("✅ Registry is valid")
                print(f"Total nodes: {validation_result.total_nodes}")
                print(f"Total connections: {validation_result.total_connections}")
                
                if validation_result.warnings:
                    print(f"\n⚠️  Warnings ({len(validation_result.warnings)}):")
                    for warning in validation_result.warnings:
                        print(f"  - {warning}")
            else:
                print("❌ Registry validation failed")
                print(f"Errors ({len(validation_result.errors)}):")
                for error in validation_result.errors:
                    print(f"  - {error}")
                
                if validation_result.warnings:
                    print(f"\nWarnings ({len(validation_result.warnings)}):")
                    for warning in validation_result.warnings:
                        print(f"  - {warning}")
                        
    except Exception as e:
        logger.error(f"Failed to validate registry: {str(e)}")
        print(f"Error validating registry: {str(e)}")


def registry_stats(args):
    """Show registry statistics."""
    try:
        registry = CatalogRegistry()
        stats = registry.get_registry_stats()
        
        if args.format == 'json':
            print(json.dumps(stats, indent=2))
        else:
            print("\nRegistry Statistics:")
            print("=" * 20)
            print(f"Total Pipelines: {stats['total_pipelines']}")
            print(f"Total Connections: {stats['total_connections']}")
            print(f"Connection Density: {stats['connection_density']:.3f}")
            
            print(f"\nFrameworks ({len(stats['frameworks'])}):")
            for framework, count in stats['frameworks'].items():
                print(f"  - {framework}: {count}")
            
            print(f"\nComplexity Levels ({len(stats['complexity_levels'])}):")
            for complexity, count in stats['complexity_levels'].items():
                print(f"  - {complexity}: {count}")
            
            print(f"\nMODS Compatibility:")
            print(f"  - Compatible: {stats['mods_compatible']}")
            print(f"  - Not Compatible: {stats['mods_incompatible']}")
            
            print(f"\nTag Categories:")
            for category, tags in stats['tag_categories'].items():
                print(f"  - {category}: {len(tags)} tags")
                
    except Exception as e:
        logger.error(f"Failed to get registry stats: {str(e)}")
        print(f"Error getting registry stats: {str(e)}")


def registry_export(args):
    """Export pipeline metadata."""
    try:
        registry = CatalogRegistry()
        
        if args.pipelines:
            pipeline_ids = [pid.strip() for pid in args.pipelines.split(',')]
            export_data = []
            
            for pipeline_id in pipeline_ids:
                node = registry.get_node(pipeline_id)
                if node:
                    export_data.append(node.to_dict())
                else:
                    logger.warning(f"Pipeline not found: {pipeline_id}")
        else:
            # Export all pipelines
            all_nodes = registry.get_all_nodes()
            export_data = [node.to_dict() for node in all_nodes]
        
        if args.format == 'json':
            output = json.dumps(export_data, indent=2)
        else:  # yaml
            try:
                import yaml
                output = yaml.dump(export_data, default_flow_style=False)
            except ImportError:
                logger.error("PyYAML not installed. Using JSON format.")
                output = json.dumps(export_data, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Exported {len(export_data)} pipelines to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        logger.error(f"Failed to export registry: {str(e)}")
        print(f"Error exporting registry: {str(e)}")


# ============================================================================
# Discovery Commands
# ============================================================================

def find_pipelines(args):
    """Find pipelines by criteria using tag-based discovery."""
    try:
        discovery = TagBasedDiscovery()
        
        # Build search criteria
        criteria = {}
        if args.tags:
            criteria['tags'] = [tag.strip() for tag in args.tags.split(',')]
        if args.framework:
            criteria['framework'] = args.framework
        if args.complexity:
            criteria['complexity'] = args.complexity
        if args.use_case:
            criteria['use_case'] = args.use_case
        if args.features:
            criteria['features'] = [feature.strip() for feature in args.features.split(',')]
        if args.mods_compatible:
            criteria['mods_compatible'] = True
        
        # Perform search
        results = discovery.search_pipelines(criteria, limit=args.limit)
        
        if args.format == 'json':
            # Convert search results to JSON-serializable format
            json_results = []
            for result in results:
                json_results.append({
                    'pipeline': result.pipeline.to_dict(),
                    'score': result.score,
                    'matched_criteria': result.matched_criteria
                })
            print(json.dumps(json_results, indent=2))
        else:
            print(f"\nSearch Results ({len(results)} found):")
            print("=" * 50)
            
            if not results:
                print("No pipelines found matching the criteria.")
                return
            
            print(f"{'ID':<25} {'TITLE':<35} {'FRAMEWORK':<12} {'SCORE':<6}")
            print('-' * 78)
            
            for result in results:
                pipeline = result.pipeline
                print(f"{pipeline.atomic_id:<25} {pipeline.title[:34]:<35} "
                      f"{pipeline.framework:<12} {result.score:.2f}")
            
            # Show search criteria
            criteria_str = []
            if args.tags:
                criteria_str.append(f"tags: {args.tags}")
            if args.framework:
                criteria_str.append(f"framework: {args.framework}")
            if args.complexity:
                criteria_str.append(f"complexity: {args.complexity}")
            if args.use_case:
                criteria_str.append(f"use_case: {args.use_case}")
            if args.features:
                criteria_str.append(f"features: {args.features}")
            if args.mods_compatible:
                criteria_str.append("mods_compatible: true")
            
            if criteria_str:
                print(f"\nSearch criteria: {', '.join(criteria_str)}")
                
    except Exception as e:
        logger.error(f"Failed to search pipelines: {str(e)}")
        print(f"Error searching pipelines: {str(e)}")


# ============================================================================
# Connection Navigation Commands
# ============================================================================

def show_connections(args):
    """Show pipeline connections."""
    try:
        registry = CatalogRegistry()
        traverser = ConnectionTraverser()
        
        node = registry.get_node(args.pipeline)
        if not node:
            print(f"Pipeline not found: {args.pipeline}")
            return
        
        connections = traverser.get_connections(args.pipeline)
        
        if args.format == 'json':
            json_connections = []
            for conn in connections:
                json_connections.append(conn.to_dict())
            print(json.dumps(json_connections, indent=2))
        else:
            print(f"\nConnections for {args.pipeline}:")
            print("=" * 40)
            
            if not connections:
                print("No connections found.")
                return
            
            # Group by connection type
            by_type = {}
            for conn in connections:
                if conn.connection_type not in by_type:
                    by_type[conn.connection_type] = []
                by_type[conn.connection_type].append(conn)
            
            for conn_type, conns in by_type.items():
                print(f"\n{conn_type.upper()} ({len(conns)}):")
                for conn in conns:
                    target_node = registry.get_node(conn.target_id)
                    if target_node:
                        print(f"  → {conn.target_id}: {target_node.title}")
                        if conn.description:
                            print(f"    {conn.description}")
                            
    except Exception as e:
        logger.error(f"Failed to show connections: {str(e)}")
        print(f"Error showing connections: {str(e)}")


def show_alternatives(args):
    """Show alternative pipelines."""
    try:
        registry = CatalogRegistry()
        traverser = ConnectionTraverser()
        
        node = registry.get_node(args.pipeline)
        if not node:
            print(f"Pipeline not found: {args.pipeline}")
            return
        
        alternatives = traverser.find_alternatives(args.pipeline)
        
        if args.format == 'json':
            json_alternatives = []
            for alt in alternatives:
                alt_node = registry.get_node(alt)
                if alt_node:
                    json_alternatives.append(alt_node.to_dict())
            print(json.dumps(json_alternatives, indent=2))
        else:
            print(f"\nAlternatives to {args.pipeline}:")
            print("=" * 40)
            
            if not alternatives:
                print("No alternatives found.")
                return
            
            print(f"{'ID':<25} {'TITLE':<35} {'FRAMEWORK':<12}")
            print('-' * 72)
            
            for alt_id in alternatives:
                alt_node = registry.get_node(alt_id)
                if alt_node:
                    print(f"{alt_node.atomic_id:<25} {alt_node.title[:34]:<35} "
                          f"{alt_node.framework:<12}")
                          
    except Exception as e:
        logger.error(f"Failed to show alternatives: {str(e)}")
        print(f"Error showing alternatives: {str(e)}")


def find_path(args):
    """Find connection path between pipelines."""
    try:
        registry = CatalogRegistry()
        traverser = ConnectionTraverser()
        
        # Validate source and target exist
        source_node = registry.get_node(args.source)
        target_node = registry.get_node(args.target)
        
        if not source_node:
            print(f"Source pipeline not found: {args.source}")
            return
        if not target_node:
            print(f"Target pipeline not found: {args.target}")
            return
        
        path = traverser.find_path(args.source, args.target)
        
        if args.format == 'json':
            if path:
                json_path = []
                for step in path:
                    step_node = registry.get_node(step)
                    if step_node:
                        json_path.append(step_node.to_dict())
                print(json.dumps(json_path, indent=2))
            else:
                print(json.dumps({"path": None, "message": "No path found"}, indent=2))
        else:
            print(f"\nPath from {args.source} to {args.target}:")
            print("=" * 50)
            
            if not path:
                print("No connection path found.")
                return
            
            for i, step in enumerate(path):
                step_node = registry.get_node(step)
                if step_node:
                    if i == 0:
                        print(f"Start: {step} - {step_node.title}")
                    elif i == len(path) - 1:
                        print(f"End:   {step} - {step_node.title}")
                    else:
                        print(f"Step {i}: {step} - {step_node.title}")
            
            print(f"\nPath length: {len(path)} steps")
            
    except Exception as e:
        logger.error(f"Failed to find path: {str(e)}")
        print(f"Error finding path: {str(e)}")


# ============================================================================
# Recommendation Commands
# ============================================================================

def recommend_pipelines(args):
    """Get pipeline recommendations."""
    try:
        engine = PipelineRecommendationEngine()
        
        if args.use_case:
            recommendations = engine.recommend_by_use_case(
                args.use_case, 
                skill_level=args.skill_level,
                limit=args.limit
            )
        elif args.next_steps:
            recommendations = engine.recommend_next_steps(
                args.next_steps,
                limit=args.limit
            )
        elif args.learning_path:
            if not args.framework:
                print("Error: --framework is required with --learning-path")
                return
            recommendations = engine.generate_learning_path(
                args.framework,
                skill_level=args.skill_level or 'beginner'
            )
        else:
            print("Error: Must specify one of --use-case, --next-steps, or --learning-path")
            return
        
        if args.format == 'json':
            json_recommendations = []
            for rec in recommendations:
                json_recommendations.append(rec.to_dict())
            print(json.dumps(json_recommendations, indent=2))
        else:
            if args.learning_path:
                print(f"\nLearning Path for {args.framework}:")
                print("=" * 40)
                
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec.pipeline_id}")
                    print(f"   Score: {rec.score:.2f}")
                    print(f"   Reason: {rec.reason}")
                    if rec.estimated_duration:
                        print(f"   Duration: {rec.estimated_duration}")
            else:
                print(f"\nRecommendations ({len(recommendations)} found):")
                print("=" * 40)
                
                if not recommendations:
                    print("No recommendations found.")
                    return
                
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec.pipeline_id}")
                    print(f"   Score: {rec.score:.2f}")
                    print(f"   Reason: {rec.reason}")
                    if rec.estimated_duration:
                        print(f"   Duration: {rec.estimated_duration}")
                        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {str(e)}")
        print(f"Error getting recommendations: {str(e)}")


# ============================================================================
# Legacy Commands (for backward compatibility)
# ============================================================================

def list_pipelines(args):
    """List all available pipelines."""
    try:
        registry = CatalogRegistry()
        nodes = registry.get_all_nodes()
        
        # Sort nodes
        if args.sort == 'id':
            nodes.sort(key=lambda n: n.atomic_id)
        elif args.sort == 'title':
            nodes.sort(key=lambda n: n.title)
        elif args.sort == 'framework':
            nodes.sort(key=lambda n: n.framework)
        elif args.sort == 'complexity':
            nodes.sort(key=lambda n: n.complexity)
        
        if args.format == 'json':
            json_nodes = [node.to_dict() for node in nodes]
            print(json.dumps(json_nodes, indent=2))
        else:
            print(f"\nAll Pipelines ({len(nodes)} total):")
            print("=" * 50)
            print(f"{'ID':<25} {'TITLE':<35} {'FRAMEWORK':<12} {'COMPLEXITY':<12}")
            print('-' * 84)
            
            for node in nodes:
                print(f"{node.atomic_id:<25} {node.title[:34]:<35} "
                      f"{node.framework:<12} {node.complexity:<12}")
                      
    except Exception as e:
        logger.error(f"Failed to list pipelines: {str(e)}")
        print(f"Error listing pipelines: {str(e)}")


def show_pipeline(args):
    """Show details for a specific pipeline."""
    try:
        registry = CatalogRegistry()
        node = registry.get_node(args.pipeline_id)
        
        if not node:
            print(f"Pipeline not found: {args.pipeline_id}")
            return
        
        if args.format == 'json':
            print(json.dumps(node.to_dict(), indent=2))
        else:
            print(f"\n{node.title}")
            print('=' * len(node.title))
            print(f"ID: {node.atomic_id}")
            print(f"Framework: {node.framework}")
            print(f"Complexity: {node.complexity}")
            print(f"Use Case: {node.use_case}")
            print(f"MODS Compatible: {'Yes' if node.mods_compatible else 'No'}")
            
            if node.features:
                print(f"Features: {', '.join(node.features)}")
            
            print(f"\nDescription:")
            print(f"Single Responsibility: {node.single_responsibility}")
            
            if node.input_interface:
                print(f"\nInput Interface:")
                for inp in node.input_interface:
                    print(f"  - {inp}")
            
            if node.output_interface:
                print(f"\nOutput Interface:")
                for out in node.output_interface:
                    print(f"  - {out}")
            
            if node.use_cases:
                print(f"\nUse Cases:")
                for use_case in node.use_cases:
                    print(f"  - {use_case}")
            
            print(f"\nMetadata:")
            print(f"  Node Count: {node.node_count}")
            print(f"  Edge Count: {node.edge_count}")
            print(f"  Independence Level: {node.independence_level}")
            print(f"  Estimated Runtime: {node.estimated_runtime}")
            print(f"  Resource Requirements: {node.resource_requirements}")
            print(f"  Skill Level: {node.skill_level}")
            
            # Show connections
            traverser = ConnectionTraverser()
            connections = traverser.get_connections(args.pipeline_id)
            if connections:
                print(f"\nConnections ({len(connections)}):")
                by_type = {}
                for conn in connections:
                    if conn.connection_type not in by_type:
                        by_type[conn.connection_type] = []
                    by_type[conn.connection_type].append(conn)
                
                for conn_type, conns in by_type.items():
                    print(f"  {conn_type}: {len(conns)} connections")
                    
    except Exception as e:
        logger.error(f"Failed to show pipeline: {str(e)}")
        print(f"Error showing pipeline: {str(e)}")


def main(args=None):
    """
    Main entry point for the Zettelkasten catalog CLI.
    
    Args:
        args: Command-line arguments (optional)
    """
    parser = argparse.ArgumentParser(description='Zettelkasten Pipeline Catalog CLI')
    parser = setup_parser(parser)
    args = parser.parse_args(args)
    
    try:
        if args.command == 'registry':
            if args.registry_command == 'validate':
                registry_validate(args)
            elif args.registry_command == 'stats':
                registry_stats(args)
            elif args.registry_command == 'export':
                registry_export(args)
            else:
                print("Available registry commands: validate, stats, export")
        elif args.command == 'find':
            find_pipelines(args)
        elif args.command == 'connections':
            show_connections(args)
        elif args.command == 'alternatives':
            show_alternatives(args)
        elif args.command == 'path':
            find_path(args)
        elif args.command == 'recommend':
            recommend_pipelines(args)
        elif args.command == 'list':
            list_pipelines(args)
        elif args.command == 'show':
            show_pipeline(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == '__main__':
    main()
