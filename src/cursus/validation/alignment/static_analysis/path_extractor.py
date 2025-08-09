"""
Specialized path extractor for analyzing path usage patterns in scripts.

Provides focused analysis of how scripts construct and use file paths,
with special attention to SageMaker container path conventions.
"""

import os
import re
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path

from ..alignment_utils import (
    PathReference, PathConstruction, FileOperation,
    normalize_path, extract_logical_name_from_path, is_sagemaker_path
)


class PathExtractor:
    """
    Specialized extractor for path usage patterns in scripts.
    
    Identifies:
    - Hardcoded path strings
    - Path construction using os.path.join()
    - Path manipulation using pathlib
    - File operations (open, read, write)
    """
    
    def __init__(self, script_content: str, script_lines: List[str]):
        """
        Initialize the path extractor.
        
        Args:
            script_content: Full content of the script
            script_lines: List of script lines for context extraction
        """
        self.script_content = script_content
        self.script_lines = script_lines
        
    def extract_hardcoded_paths(self) -> List[str]:
        """Find all hardcoded path strings."""
        hardcoded_paths = []
        
        # Regex patterns for common path formats
        path_patterns = [
            # SageMaker paths
            r'["\'](/opt/ml/[^"\']*)["\']',
            # Absolute Unix paths
            r'["\'](/[a-zA-Z0-9_/.-]+)["\']',
            # Relative paths with ./
            r'["\'](\./[a-zA-Z0-9_/.-]+)["\']',
            # Relative paths with ../
            r'["\'](\.\./[a-zA-Z0-9_/.-]+)["\']',
            # Windows paths
            r'["\']([A-Za-z]:\\[^"\']*)["\']',
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, self.script_content)
            for match in matches:
                if self._is_likely_path(match):
                    hardcoded_paths.append(match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in hardcoded_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        
        return unique_paths
    
    def extract_path_constructions(self) -> List[PathConstruction]:
        """Find all dynamic path construction patterns."""
        constructions = []
        
        # Pattern for os.path.join calls
        join_pattern = r'os\.path\.join\s*\(\s*([^)]+)\s*\)'
        
        for line_num, line in enumerate(self.script_lines, 1):
            matches = re.finditer(join_pattern, line)
            for match in matches:
                args_str = match.group(1)
                parts = self._parse_join_arguments(args_str)
                
                if parts:
                    # Try to construct the path if all parts are strings
                    try:
                        constructed_path = os.path.join(*[p for p in parts if isinstance(p, str)])
                        base_path = parts[0] if parts else ""
                        
                        construction = PathConstruction(
                            base_path=str(base_path),
                            construction_parts=parts,
                            line_number=line_num,
                            context=self._get_line_context(line_num),
                            method='os.path.join'
                        )
                        constructions.append(construction)
                    except (TypeError, ValueError):
                        # Skip if path construction fails
                        pass
        
        # Pattern for pathlib Path operations
        pathlib_patterns = [
            r'Path\s*\(\s*([^)]+)\s*\)',
            r'PurePath\s*\(\s*([^)]+)\s*\)',
        ]
        
        for pattern in pathlib_patterns:
            for line_num, line in enumerate(self.script_lines, 1):
                matches = re.finditer(pattern, line)
                for match in matches:
                    args_str = match.group(1)
                    parts = self._parse_path_arguments(args_str)
                    
                    if parts:
                        construction = PathConstruction(
                            base_path=str(parts[0]) if parts else "",
                            construction_parts=parts,
                            line_number=line_num,
                            context=self._get_line_context(line_num),
                            method='pathlib.Path'
                        )
                        constructions.append(construction)
        
        return constructions
    
    def extract_file_operations(self) -> List[FileOperation]:
        """Find all file read/write operations."""
        operations = []
        
        # Pattern for open() calls
        open_pattern = r'open\s*\(\s*([^,)]+)(?:,\s*["\']([rwab+]+)["\'])?\s*[^)]*\)'
        
        for line_num, line in enumerate(self.script_lines, 1):
            matches = re.finditer(open_pattern, line)
            for match in matches:
                file_path_expr = match.group(1).strip()
                mode = match.group(2) if match.group(2) else 'r'
                
                # Try to extract the actual path
                file_path = self._extract_path_from_expression(file_path_expr)
                
                if file_path:
                    operation_type = self._determine_operation_type(mode)
                    
                    operation = FileOperation(
                        file_path=file_path,
                        operation_type=operation_type,
                        line_number=line_num,
                        context=self._get_line_context(line_num),
                        mode=mode
                    )
                    operations.append(operation)
        
        # Pattern for with open() statements
        with_open_pattern = r'with\s+open\s*\(\s*([^,)]+)(?:,\s*["\']([rwab+]+)["\'])?\s*[^)]*\)\s*as'
        
        for line_num, line in enumerate(self.script_lines, 1):
            matches = re.finditer(with_open_pattern, line)
            for match in matches:
                file_path_expr = match.group(1).strip()
                mode = match.group(2) if match.group(2) else 'r'
                
                file_path = self._extract_path_from_expression(file_path_expr)
                
                if file_path:
                    operation_type = self._determine_operation_type(mode)
                    
                    operation = FileOperation(
                        file_path=file_path,
                        operation_type=operation_type,
                        line_number=line_num,
                        context=self._get_line_context(line_num),
                        mode=mode
                    )
                    operations.append(operation)
        
        return operations
    
    def analyze_sagemaker_path_usage(self) -> Dict[str, List[str]]:
        """Analyze SageMaker-specific path usage patterns."""
        sagemaker_analysis = {
            'input_paths': [],
            'output_paths': [],
            'model_paths': [],
            'other_paths': []
        }
        
        all_paths = self.extract_hardcoded_paths()
        
        for path in all_paths:
            if is_sagemaker_path(path):
                if '/opt/ml/processing/input/' in path or '/opt/ml/input/data/' in path:
                    sagemaker_analysis['input_paths'].append(path)
                elif '/opt/ml/processing/output/' in path or '/opt/ml/output/' in path:
                    sagemaker_analysis['output_paths'].append(path)
                elif '/opt/ml/model' in path:
                    sagemaker_analysis['model_paths'].append(path)
                else:
                    sagemaker_analysis['other_paths'].append(path)
        
        return sagemaker_analysis
    
    def extract_logical_names_from_paths(self) -> Dict[str, str]:
        """Extract logical names from SageMaker paths."""
        logical_names = {}
        
        all_paths = self.extract_hardcoded_paths()
        
        for path in all_paths:
            logical_name = extract_logical_name_from_path(path)
            if logical_name:
                logical_names[logical_name] = path
        
        return logical_names
    
    def find_path_inconsistencies(self) -> List[Dict[str, Any]]:
        """Find potential path usage inconsistencies."""
        inconsistencies = []
        
        # Check for mixed path construction methods
        hardcoded = set(self.extract_hardcoded_paths())
        constructions = self.extract_path_constructions()
        
        # Look for the same logical path constructed in different ways
        constructed_paths = set()
        for construction in constructions:
            try:
                if construction.method == 'os.path.join':
                    path = os.path.join(*[str(p) for p in construction.construction_parts if isinstance(p, str)])
                    constructed_paths.add(normalize_path(path))
            except:
                pass
        
        # Find paths that appear both hardcoded and constructed
        for hardcoded_path in hardcoded:
            normalized_hardcoded = normalize_path(hardcoded_path)
            if normalized_hardcoded in constructed_paths:
                inconsistencies.append({
                    'type': 'mixed_construction',
                    'path': hardcoded_path,
                    'issue': 'Path appears both hardcoded and dynamically constructed',
                    'recommendation': 'Use consistent path construction method'
                })
        
        # Check for non-SageMaker paths in SageMaker scripts
        non_sagemaker_paths = [p for p in hardcoded if not is_sagemaker_path(p)]
        if non_sagemaker_paths:
            for path in non_sagemaker_paths:
                if not path.startswith(('./', '../', '/tmp')):  # Allow some common patterns
                    inconsistencies.append({
                        'type': 'non_sagemaker_path',
                        'path': path,
                        'issue': 'Non-SageMaker path found in processing script',
                        'recommendation': 'Use SageMaker container paths (/opt/ml/...)'
                    })
        
        return inconsistencies
    
    def _is_likely_path(self, string_value: str) -> bool:
        """Check if a string is likely to be a file path."""
        # Skip very short strings
        if len(string_value) < 3:
            return False
        
        # Skip strings that look like URLs
        if string_value.startswith(('http://', 'https://', 'ftp://', 'file://')):
            return False
        
        # Skip strings that look like environment variable names
        if string_value.isupper() and '_' in string_value and '/' not in string_value:
            return False
        
        # Check for path indicators
        path_indicators = [
            '/',  # Unix path separator
            '\\',  # Windows path separator
            '/opt/ml/',  # SageMaker paths
            '/tmp/',
            '/var/',
            '/home/',
            '/usr/',
            './',
            '../',
        ]
        
        for indicator in path_indicators:
            if indicator in string_value:
                return True
        
        # Check for file extensions
        if '.' in string_value:
            extension = string_value.split('.')[-1]
            if len(extension) <= 5 and extension.isalnum():
                return True
        
        return False
    
    def _parse_join_arguments(self, args_str: str) -> List[str]:
        """Parse arguments from os.path.join call."""
        parts = []
        
        # Simple parsing - split by comma and clean up
        raw_parts = args_str.split(',')
        
        for part in raw_parts:
            part = part.strip()
            
            # Remove quotes if present
            if (part.startswith('"') and part.endswith('"')) or \
               (part.startswith("'") and part.endswith("'")):
                part = part[1:-1]
                parts.append(part)
            else:
                # Variable or expression - keep as is for now
                parts.append(part)
        
        return parts
    
    def _parse_path_arguments(self, args_str: str) -> List[str]:
        """Parse arguments from pathlib Path constructor."""
        # Similar to join arguments but for pathlib
        return self._parse_join_arguments(args_str)
    
    def _extract_path_from_expression(self, expression: str) -> Optional[str]:
        """Extract path string from a Python expression."""
        expression = expression.strip()
        
        # Handle quoted strings
        if (expression.startswith('"') and expression.endswith('"')) or \
           (expression.startswith("'") and expression.endswith("'")):
            return expression[1:-1]
        
        # Handle f-strings (basic case)
        if expression.startswith('f"') or expression.startswith("f'"):
            # For now, just return the literal part
            quote_char = expression[1]
            if expression.endswith(quote_char):
                return expression[2:-1]
        
        # Handle variables - we can't resolve these statically
        # but we can check if they look like path variables
        if any(keyword in expression.lower() for keyword in ['path', 'dir', 'file']):
            return expression  # Return as-is for now
        
        return None
    
    def _determine_operation_type(self, mode: str) -> str:
        """Determine operation type from file mode."""
        if 'w' in mode or 'a' in mode:
            return 'write'
        elif 'r' in mode:
            return 'read'
        else:
            return 'unknown'
    
    def _get_line_context(self, line_number: int, context_lines: int = 2) -> str:
        """Get context around a specific line number."""
        start = max(0, line_number - context_lines - 1)
        end = min(len(self.script_lines), line_number + context_lines)
        context_lines_list = self.script_lines[start:end]
        
        # Mark the target line
        target_index = line_number - start - 1
        if 0 <= target_index < len(context_lines_list):
            context_lines_list[target_index] = f">>> {context_lines_list[target_index]}"
        
        return '\n'.join(context_lines_list)
    
    def get_path_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of path usage in the script."""
        hardcoded_paths = self.extract_hardcoded_paths()
        constructions = self.extract_path_constructions()
        file_operations = self.extract_file_operations()
        sagemaker_analysis = self.analyze_sagemaker_path_usage()
        logical_names = self.extract_logical_names_from_paths()
        inconsistencies = self.find_path_inconsistencies()
        
        return {
            'hardcoded_paths': hardcoded_paths,
            'path_constructions': len(constructions),
            'file_operations': len(file_operations),
            'sagemaker_paths': {
                'input_paths': len(sagemaker_analysis['input_paths']),
                'output_paths': len(sagemaker_analysis['output_paths']),
                'model_paths': len(sagemaker_analysis['model_paths']),
                'other_paths': len(sagemaker_analysis['other_paths'])
            },
            'logical_names': list(logical_names.keys()),
            'inconsistencies': len(inconsistencies),
            'total_paths_found': len(hardcoded_paths) + len(constructions)
        }
