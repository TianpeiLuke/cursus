"""
Command-line interface for the pipeline catalog.

This module provides CLI commands for working with the pipeline catalog, including
browsing, searching, and managing the catalog.

Example usage:
    cursus catalog list
    cursus catalog search --framework xgboost --feature training
    cursus catalog show xgboost-simple
    cursus catalog generate xgboost-simple --output my_pipeline.py
    cursus catalog update-index
"""

import os
import json
import shutil
import argparse
import textwrap
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

from src.cursus.pipeline_catalog.utils import (
    load_index, 
    get_pipeline_by_id, 
    filter_pipelines,
    get_pipeline_path,
    get_all_frameworks,
    get_all_features
)
from src.cursus.pipeline_catalog.indexer import CatalogIndexer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def setup_parser(parser):
    """
    Set up the argument parser for the catalog CLI.
    
    Args:
        parser: The argparse parser to configure
    """
    subparsers = parser.add_subparsers(dest='command', help='Catalog command')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all available pipelines')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table',
                           help='Output format')
    list_parser.add_argument('--sort', choices=['id', 'name', 'framework', 'complexity'], 
                            default='id', help='Sort field')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search pipelines by criteria')
    search_parser.add_argument('--framework', help='Filter by framework (e.g., xgboost, pytorch)')
    search_parser.add_argument('--complexity', choices=['simple', 'standard', 'advanced'], 
                              help='Filter by complexity')
    search_parser.add_argument('--feature', action='append', dest='features',
                              help='Filter by feature (can be used multiple times)')
    search_parser.add_argument('--tag', action='append', dest='tags',
                              help='Filter by tag (can be used multiple times)')
    search_parser.add_argument('--format', choices=['table', 'json'], default='table',
                             help='Output format')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show details for a pipeline')
    show_parser.add_argument('pipeline_id', help='ID of the pipeline to show')
    show_parser.add_argument('--format', choices=['text', 'json'], default='text',
                           help='Output format')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a pipeline from template')
    generate_parser.add_argument('pipeline_id', help='ID of the pipeline to generate')
    generate_parser.add_argument('--output', '-o', required=True, help='Output file path')
    generate_parser.add_argument('--rename', help='New name for the pipeline')
    
    # Update index command
    update_parser = subparsers.add_parser('update-index', help='Update the pipeline catalog index')
    update_parser.add_argument('--force', action='store_true', 
                             help='Force full regeneration of the index')
    update_parser.add_argument('--validate-only', action='store_true',
                             help='Only validate the index without updating')
                             
    # Validate index command
    validate_parser = subparsers.add_parser('validate', help='Validate the pipeline catalog index')
    
    return parser


def list_pipelines(args):
    """
    List all available pipelines.
    
    Args:
        args: Command-line arguments
    """
    try:
        index = load_index()
        pipelines = index.get('pipelines', [])
        
        # Sort pipelines
        sort_field = args.sort
        pipelines.sort(key=lambda p: p.get(sort_field, ''))
        
        if args.format == 'json':
            print(json.dumps(pipelines, indent=2))
        else:
            # Table format
            print(f"\n{'ID':<20} {'NAME':<30} {'FRAMEWORK':<12} {'COMPLEXITY':<10}")
            print('-' * 72)
            
            for pipeline in pipelines:
                print(f"{pipeline.get('id', ''):<20} {pipeline.get('name', ''):<30} "
                      f"{pipeline.get('framework', ''):<12} {pipeline.get('complexity', ''):<10}")
                      
            print(f"\nTotal: {len(pipelines)} pipelines")
    except Exception as e:
        logger.error(f"Failed to list pipelines: {str(e)}")


def search_pipelines(args):
    """
    Search pipelines by criteria.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Apply filters
        pipelines = filter_pipelines(
            framework=args.framework,
            complexity=args.complexity,
            features=args.features,
            tags=args.tags
        )
        
        if args.format == 'json':
            print(json.dumps(pipelines, indent=2))
        else:
            # Table format
            print(f"\nSearch results:")
            print(f"{'ID':<20} {'NAME':<30} {'FRAMEWORK':<12} {'COMPLEXITY':<10}")
            print('-' * 72)
            
            for pipeline in pipelines:
                print(f"{pipeline.get('id', ''):<20} {pipeline.get('name', ''):<30} "
                      f"{pipeline.get('framework', ''):<12} {pipeline.get('complexity', ''):<10}")
                      
            print(f"\nTotal: {len(pipelines)} pipelines found")
            
            # Show applied filters
            filters = []
            if args.framework:
                filters.append(f"framework={args.framework}")
            if args.complexity:
                filters.append(f"complexity={args.complexity}")
            if args.features:
                filters.append(f"features={args.features}")
            if args.tags:
                filters.append(f"tags={args.tags}")
                
            if filters:
                print(f"Applied filters: {', '.join(filters)}")
                
    except Exception as e:
        logger.error(f"Failed to search pipelines: {str(e)}")


def show_pipeline(args):
    """
    Show details for a specific pipeline.
    
    Args:
        args: Command-line arguments
    """
    try:
        pipeline = get_pipeline_by_id(args.pipeline_id)
        
        if not pipeline:
            logger.error(f"Pipeline not found: {args.pipeline_id}")
            return
            
        if args.format == 'json':
            print(json.dumps(pipeline, indent=2))
        else:
            # Text format
            print(f"\n{pipeline.get('name', 'Unnamed Pipeline')}")
            print('=' * len(pipeline.get('name', 'Unnamed Pipeline')))
            print(f"ID: {pipeline.get('id', '')}")
            print(f"Framework: {pipeline.get('framework', '')}")
            print(f"Complexity: {pipeline.get('complexity', '')}")
            
            features = pipeline.get('features', [])
            if features:
                print(f"Features: {', '.join(features)}")
                
            tags = pipeline.get('tags', [])
            if tags:
                print(f"Tags: {', '.join(tags)}")
                
            path = pipeline.get('path', '')
            if path:
                print(f"Path: {path}")
                
            description = pipeline.get('description', '')
            if description:
                print("\nDescription:")
                print(textwrap.fill(description, width=80))
                
            # Try to extract usage example from the source file
            try:
                pipeline_path = get_pipeline_path(args.pipeline_id)
                if pipeline_path and pipeline_path.exists():
                    with open(pipeline_path, 'r') as f:
                        content = f.read()
                    
                    # Extract example block
                    if "Example:" in content and "```python" in content:
                        start_idx = content.find("```python", content.find("Example:"))
                        if start_idx > -1:
                            end_idx = content.find("```", start_idx + 10)
                            if end_idx > -1:
                                example = content[start_idx+10:end_idx].strip()
                                print("\nUsage Example:")
                                print(example)
            except Exception as e:
                logger.debug(f"Failed to extract usage example: {str(e)}")
                
    except Exception as e:
        logger.error(f"Failed to show pipeline details: {str(e)}")


def generate_pipeline(args):
    """
    Generate a pipeline from a template.
    
    Args:
        args: Command-line arguments
    """
    try:
        pipeline = get_pipeline_by_id(args.pipeline_id)
        
        if not pipeline:
            logger.error(f"Pipeline not found: {args.pipeline_id}")
            return
        
        # Get the source path
        source_path = get_pipeline_path(args.pipeline_id)
        if not source_path or not source_path.exists():
            logger.error(f"Pipeline source file not found: {args.pipeline_id}")
            return
            
        # Create output path
        output_path = Path(args.output)
        output_dir = output_path.parent
        
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            
        # Copy the file
        shutil.copy2(source_path, output_path)
        logger.info(f"Generated pipeline file: {output_path}")
        
        # Rename the pipeline if requested
        if args.rename:
            with open(output_path, 'r') as f:
                content = f.read()
                
            # Get first line of docstring which is typically the name
            if '"""' in content:
                start_idx = content.find('"""') + 3
                end_idx = content.find('\n', start_idx)
                
                if end_idx > start_idx:
                    old_name = content[start_idx:end_idx].strip()
                    new_content = content.replace(old_name, args.rename, 1)
                    
                    with open(output_path, 'w') as f:
                        f.write(new_content)
                        
                    logger.info(f"Renamed pipeline to: {args.rename}")
                    
    except Exception as e:
        logger.error(f"Failed to generate pipeline: {str(e)}")


def update_index(args):
    """
    Update the pipeline catalog index.
    
    Args:
        args: Command-line arguments
    """
    try:
        indexer = CatalogIndexer()
        
        if args.validate_only:
            # Only validate the existing index
            try:
                with open(indexer.index_path, "r") as f:
                    index = json.load(f)
                
                is_valid, issues = indexer.validate_index(index)
                if is_valid:
                    logger.info("Index validation successful")
                    print("The pipeline catalog index is valid.")
                else:
                    logger.warning("Index validation failed:")
                    print("The pipeline catalog index has validation issues:")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
                        print(f"  - {issue}")
            except Exception as e:
                logger.error(f"Failed to validate index: {str(e)}")
                print(f"Error validating index: {str(e)}")
        else:
            # Generate and update index
            if args.force:
                # Generate new index from scratch
                new_index = indexer.generate_index()
                indexer.save_index(new_index)
                logger.info("Regenerated index from scratch")
                print(f"Successfully regenerated pipeline catalog index with {len(new_index['pipelines'])} entries.")
            else:
                # Update existing index
                indexer.update_index()
                logger.info("Updated index")
                
                # Load the updated index to show count
                try:
                    with open(indexer.index_path, "r") as f:
                        index = json.load(f)
                    print(f"Successfully updated pipeline catalog index with {len(index['pipelines'])} entries.")
                except:
                    print("Successfully updated pipeline catalog index.")
                    
    except Exception as e:
        logger.error(f"Failed to update index: {str(e)}")
        print(f"Error updating index: {str(e)}")


def validate_index(args):
    """
    Validate the pipeline catalog index.
    
    Args:
        args: Command-line arguments
    """
    try:
        indexer = CatalogIndexer()
        
        try:
            with open(indexer.index_path, "r") as f:
                index = json.load(f)
            
            is_valid, issues = indexer.validate_index(index)
            if is_valid:
                logger.info("Index validation successful")
                print("The pipeline catalog index is valid.")
                print(f"Total entries: {len(index['pipelines'])}")
                
                # Print some stats
                frameworks = {}
                complexities = {}
                features = set()
                for pipeline in index.get('pipelines', []):
                    # Count by framework
                    framework = pipeline.get('framework', 'unknown')
                    frameworks[framework] = frameworks.get(framework, 0) + 1
                    
                    # Count by complexity
                    complexity = pipeline.get('complexity', 'unknown')
                    complexities[complexity] = complexities.get(complexity, 0) + 1
                    
                    # Collect all features
                    features.update(pipeline.get('features', []))
                
                print("\nFrameworks:")
                for framework, count in sorted(frameworks.items()):
                    print(f"  - {framework}: {count}")
                
                print("\nComplexities:")
                for complexity, count in sorted(complexities.items()):
                    print(f"  - {complexity}: {count}")
                
                print("\nFeatures available:")
                for feature in sorted(features):
                    print(f"  - {feature}")
                
            else:
                logger.warning("Index validation failed:")
                print("The pipeline catalog index has validation issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                    print(f"  - {issue}")
                    
                print("\nRun 'cursus catalog update-index --force' to regenerate the index.")
        except FileNotFoundError:
            logger.error("Index file not found")
            print("The pipeline catalog index file does not exist.")
            print("Run 'cursus catalog update-index' to generate it.")
        except json.JSONDecodeError:
            logger.error("Invalid index file format")
            print("The pipeline catalog index file is not valid JSON.")
            print("Run 'cursus catalog update-index --force' to regenerate it.")
        except Exception as e:
            logger.error(f"Failed to validate index: {str(e)}")
            print(f"Error validating index: {str(e)}")
                
    except Exception as e:
        logger.error(f"Failed to validate index: {str(e)}")
        print(f"Error validating index: {str(e)}")


def main(args=None):
    """
    Main entry point for the catalog CLI.
    
    Args:
        args: Command-line arguments (optional)
    """
    parser = argparse.ArgumentParser(description='Pipeline Catalog CLI')
    parser = setup_parser(parser)
    args = parser.parse_args(args)
    
    if args.command == 'list':
        list_pipelines(args)
    elif args.command == 'search':
        search_pipelines(args)
    elif args.command == 'show':
        show_pipeline(args)
    elif args.command == 'generate':
        generate_pipeline(args)
    elif args.command == 'update-index':
        update_index(args)
    elif args.command == 'validate':
        validate_index(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
