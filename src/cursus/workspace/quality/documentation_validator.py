"""
Documentation Quality Validation System for Phase 3 Implementation

This module provides comprehensive documentation quality assessment as part of the
workspace-aware redundancy reduction Phase 3: Quality Assurance and Validation.

Implements anti-redundancy principles by enhancing existing validation capabilities
rather than creating duplicate documentation checking systems.
"""

import os
import re
import ast
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..validation.base_validation_result import BaseValidationResult


class DocumentationType(Enum):
    """Types of documentation to validate."""
    API_REFERENCE = "api_reference"
    QUICK_START = "quick_start"
    DEVELOPER_GUIDE = "developer_guide"
    DESIGN_DOCUMENT = "design_document"
    README = "readme"
    DOCSTRING = "docstring"


@dataclass
class DocumentationMetrics:
    """Metrics for documentation quality assessment."""
    completeness_score: float  # 0.0 - 1.0
    clarity_score: float       # 0.0 - 1.0
    accuracy_score: float      # 0.0 - 1.0
    coverage_score: float      # 0.0 - 1.0
    consistency_score: float   # 0.0 - 1.0
    accessibility_score: float # 0.0 - 1.0
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall documentation quality score."""
        weights = {
            'completeness': 0.25,
            'clarity': 0.20,
            'accuracy': 0.20,
            'coverage': 0.15,
            'consistency': 0.10,
            'accessibility': 0.10
        }
        
        return (
            self.completeness_score * weights['completeness'] +
            self.clarity_score * weights['clarity'] +
            self.accuracy_score * weights['accuracy'] +
            self.coverage_score * weights['coverage'] +
            self.consistency_score * weights['consistency'] +
            self.accessibility_score * weights['accessibility']
        )


class DocumentationValidationResult:
    """Documentation validation result with Phase 3 compliance checking."""
    
    def __init__(self, 
                 validation_type: str = "documentation_quality",
                 metrics: Optional[DocumentationMetrics] = None,
                 issues: Optional[List[str]] = None,
                 recommendations: Optional[List[str]] = None,
                 workspace_path: Optional[Path] = None,
                 **kwargs):
        # Set up the fields
        self.validation_type = validation_type
        self.metrics = metrics or DocumentationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.issues = issues or []
        self.recommendations = recommendations or []
        self.workspace_path = workspace_path or Path('.')
        
        # Set validation status based on Phase 3 quality thresholds
        self.is_valid = self.metrics.overall_score >= 0.85  # 85% threshold for docs
        self.quality_score = self.metrics.overall_score
        self.success = self.is_valid  # Update success based on validation
        
        # Additional BaseValidationResult-like fields for compatibility
        self.timestamp = kwargs.get('timestamp', None)
        self.messages = kwargs.get('messages', [])
        self.warnings = kwargs.get('warnings', [])
        self.errors = kwargs.get('errors', [])
        
    def meets_phase3_requirements(self) -> bool:
        """Check if documentation meets Phase 3 quality requirements."""
        return (
            self.metrics.overall_score >= 0.85 and
            self.metrics.completeness_score >= 0.80 and
            self.metrics.clarity_score >= 0.80 and
            len(self.issues) <= 5  # Maximum 5 critical issues
        )


class DocumentationQualityValidator:
    """
    Documentation quality validation system for Phase 3 implementation.
    
    Provides comprehensive documentation assessment including completeness,
    clarity, accuracy, coverage, consistency, and accessibility validation.
    """
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.validation_patterns = self._initialize_validation_patterns()
        
    def _initialize_validation_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation patterns for different documentation types."""
        return {
            'api_reference': {
                'required_sections': ['Parameters', 'Returns', 'Examples', 'Raises'],
                'quality_indicators': [
                    r'```python',  # Code examples
                    r'Args:',      # Parameter documentation
                    r'Returns:',   # Return documentation
                    r'Raises:',    # Exception documentation
                ],
                'completeness_weight': 0.3
            },
            'quick_start': {
                'required_sections': ['Installation', 'Basic Usage', 'Examples'],
                'quality_indicators': [
                    r'```',        # Code blocks
                    r'pip install', # Installation commands
                    r'import',     # Import examples
                ],
                'completeness_weight': 0.25
            },
            'developer_guide': {
                'required_sections': ['Overview', 'Architecture', 'Best Practices'],
                'quality_indicators': [
                    r'##',         # Section headers
                    r'```',        # Code examples
                    r'Note:',      # Important notes
                ],
                'completeness_weight': 0.2
            },
            'design_document': {
                'required_sections': ['Problem', 'Solution', 'Implementation'],
                'quality_indicators': [
                    r'##',         # Section structure
                    r'```',        # Code/config examples
                    r'Decision:',  # Design decisions
                ],
                'completeness_weight': 0.15
            },
            'readme': {
                'required_sections': ['Description', 'Installation', 'Usage'],
                'quality_indicators': [
                    r'#',          # Headers
                    r'```',        # Code blocks
                    r'https?://',  # Links
                ],
                'completeness_weight': 0.1
            }
        }
    
    def validate_documentation_file(self, file_path: str, doc_type: DocumentationType) -> DocumentationValidationResult:
        """Validate a single documentation file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metrics = self._assess_documentation_metrics(content, doc_type)
            issues = self._identify_documentation_issues(content, doc_type)
            recommendations = self._generate_recommendations(metrics, issues)
            
            return DocumentationValidationResult(
                validation_type=f"documentation_{doc_type.value}",
                metrics=metrics,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            return DocumentationValidationResult(
                validation_type=f"documentation_{doc_type.value}",
                issues=[f"Failed to validate documentation: {str(e)}"]
            )
    
    def validate_docstring_coverage(self, source_dir: str) -> DocumentationValidationResult:
        """Validate docstring coverage for Python source files."""
        coverage_data = []
        total_functions = 0
        documented_functions = 0
        
        for py_file in Path(source_dir).rglob("*.py"):
            if py_file.name.startswith('__'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                file_coverage = self._analyze_docstring_coverage(tree)
                coverage_data.append((str(py_file), file_coverage))
                total_functions += file_coverage['total']
                documented_functions += file_coverage['documented']
                
            except Exception as e:
                coverage_data.append((str(py_file), {'error': str(e)}))
        
        coverage_ratio = documented_functions / max(total_functions, 1)
        
        metrics = DocumentationMetrics(
            completeness_score=coverage_ratio,
            clarity_score=0.8 if coverage_ratio > 0.7 else 0.6,
            accuracy_score=0.9,  # Assume high accuracy for existing docstrings
            coverage_score=coverage_ratio,
            consistency_score=0.8,  # Based on docstring format consistency
            accessibility_score=0.7   # Standard docstring accessibility
        )
        
        issues = []
        if coverage_ratio < 0.8:
            issues.append(f"Low docstring coverage: {coverage_ratio:.1%} (target: 80%)")
        
        return DocumentationValidationResult(
            validation_type="docstring_coverage",
            metrics=metrics,
            issues=issues
        )
    
    def _assess_documentation_metrics(self, content: str, doc_type: DocumentationType) -> DocumentationMetrics:
        """Assess comprehensive documentation quality metrics."""
        patterns = self.validation_patterns.get(doc_type.value, {})
        
        # Completeness assessment
        completeness = self._assess_completeness(content, patterns)
        
        # Clarity assessment (readability, structure)
        clarity = self._assess_clarity(content)
        
        # Accuracy assessment (code examples, links)
        accuracy = self._assess_accuracy(content)
        
        # Coverage assessment (topic coverage)
        coverage = self._assess_coverage(content, doc_type)
        
        # Consistency assessment (formatting, style)
        consistency = self._assess_consistency(content)
        
        # Accessibility assessment (language, structure)
        accessibility = self._assess_accessibility(content)
        
        return DocumentationMetrics(
            completeness_score=completeness,
            clarity_score=clarity,
            accuracy_score=accuracy,
            coverage_score=coverage,
            consistency_score=consistency,
            accessibility_score=accessibility
        )
    
    def _assess_completeness(self, content: str, patterns: Dict[str, Any]) -> float:
        """Assess documentation completeness based on required sections."""
        required_sections = patterns.get('required_sections', [])
        if not required_sections:
            return 0.8  # Default score for unknown types
        
        found_sections = 0
        for section in required_sections:
            if re.search(rf'\b{re.escape(section)}\b', content, re.IGNORECASE):
                found_sections += 1
        
        return found_sections / len(required_sections)
    
    def _assess_clarity(self, content: str) -> float:
        """Assess documentation clarity and readability."""
        score = 0.5  # Base score
        
        # Check for clear structure (headers)
        if re.search(r'^#+\s+', content, re.MULTILINE):
            score += 0.2
        
        # Check for examples
        if re.search(r'```', content):
            score += 0.2
        
        # Check for clear language indicators
        if re.search(r'\b(Note|Important|Warning|Example)\b', content):
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_accuracy(self, content: str) -> float:
        """Assess documentation accuracy (links, code examples)."""
        score = 0.8  # Assume good accuracy by default
        
        # Check for broken markdown links (basic check)
        broken_links = re.findall(r'\[([^\]]+)\]\(\)', content)
        if broken_links:
            score -= 0.2
        
        # Check for incomplete code blocks
        code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', content, re.DOTALL)
        for lang, code in code_blocks:
            if not code.strip():
                score -= 0.1
        
        return max(score, 0.0)
    
    def _assess_coverage(self, content: str, doc_type: DocumentationType) -> float:
        """Assess topic coverage based on documentation type."""
        # Basic coverage assessment based on content length and structure
        word_count = len(content.split())
        
        coverage_thresholds = {
            DocumentationType.API_REFERENCE: 500,
            DocumentationType.QUICK_START: 300,
            DocumentationType.DEVELOPER_GUIDE: 800,
            DocumentationType.DESIGN_DOCUMENT: 600,
            DocumentationType.README: 200
        }
        
        threshold = coverage_thresholds.get(doc_type, 400)
        coverage_ratio = min(word_count / threshold, 1.0)
        
        return coverage_ratio
    
    def _assess_consistency(self, content: str) -> float:
        """Assess documentation consistency in formatting and style."""
        score = 0.7  # Base consistency score
        
        # Check header consistency
        headers = re.findall(r'^(#+)\s+', content, re.MULTILINE)
        if headers and len(set(len(h) for h in headers)) <= 3:
            score += 0.1
        
        # Check code block consistency
        code_blocks = re.findall(r'```(\w*)', content)
        if code_blocks and len(set(code_blocks)) <= 3:
            score += 0.1
        
        # Check bullet point consistency
        bullets = re.findall(r'^(\s*[-*+])\s+', content, re.MULTILINE)
        if bullets and len(set(bullets)) <= 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_accessibility(self, content: str) -> float:
        """Assess documentation accessibility and ease of understanding."""
        score = 0.6  # Base accessibility score
        
        # Check for clear navigation (table of contents, headers)
        if re.search(r'#+.*Table of Contents', content, re.IGNORECASE):
            score += 0.2
        
        # Check for explanatory text before code examples
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        explained_blocks = 0
        for i, block in enumerate(code_blocks):
            # Look for explanatory text before code block
            if i == 0 or re.search(r'\w+.*\n\s*```', content):
                explained_blocks += 1
        
        if code_blocks and explained_blocks / len(code_blocks) > 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_docstring_coverage(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze docstring coverage for an AST tree."""
        total = 0
        documented = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total += 1
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    documented += 1
        
        return {'total': total, 'documented': documented}
    
    def _identify_documentation_issues(self, content: str, doc_type: DocumentationType) -> List[str]:
        """Identify specific documentation quality issues."""
        issues = []
        
        # Check for empty sections
        if re.search(r'#+\s+\w+\s*\n\s*\n#+', content):
            issues.append("Empty sections detected")
        
        # Check for missing code examples in API docs
        if doc_type == DocumentationType.API_REFERENCE and not re.search(r'```python', content):
            issues.append("Missing Python code examples in API documentation")
        
        # Check for broken internal links
        internal_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        for text, link in internal_links:
            if link.startswith('#') and not re.search(rf'#+.*{re.escape(link[1:].replace("-", ".*"))}', content, re.IGNORECASE):
                issues.append(f"Broken internal link: {link}")
        
        # Check for outdated content indicators
        if re.search(r'\b(TODO|FIXME|XXX)\b', content):
            issues.append("Contains TODO/FIXME markers indicating incomplete content")
        
        return issues
    
    def _generate_recommendations(self, metrics: DocumentationMetrics, issues: List[str]) -> List[str]:
        """Generate improvement recommendations based on metrics and issues."""
        recommendations = []
        
        if metrics.completeness_score < 0.8:
            recommendations.append("Add missing required sections to improve completeness")
        
        if metrics.clarity_score < 0.7:
            recommendations.append("Improve structure with clear headers and examples")
        
        if metrics.coverage_score < 0.6:
            recommendations.append("Expand content to provide more comprehensive coverage")
        
        if metrics.consistency_score < 0.7:
            recommendations.append("Standardize formatting and style for better consistency")
        
        if metrics.accessibility_score < 0.7:
            recommendations.append("Add explanatory text and improve navigation structure")
        
        if len(issues) > 3:
            recommendations.append("Address identified issues to improve overall quality")
        
        return recommendations
    
    def validate_workspace_documentation(self) -> Dict[str, DocumentationValidationResult]:
        """Validate all documentation in the workspace."""
        results = {}
        
        # Validate key documentation files
        doc_files = {
            'api_reference': 'slipbox/workspace/workspace_api_reference.md',
            'quick_start': 'slipbox/workspace/workspace_quick_start.md',
            'developer_guide': 'slipbox/0_developer_guide/README.md',
            'readme': 'README.md'
        }
        
        for doc_name, file_path in doc_files.items():
            full_path = self.workspace_root / file_path
            if full_path.exists():
                doc_type = DocumentationType(doc_name)
                results[doc_name] = self.validate_documentation_file(str(full_path), doc_type)
        
        # Validate docstring coverage
        src_dir = self.workspace_root / 'src' / 'cursus'
        if src_dir.exists():
            results['docstring_coverage'] = self.validate_docstring_coverage(str(src_dir))
        
        return results
    
    def generate_documentation_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive documentation quality report for Phase 3."""
        validation_results = self.validate_workspace_documentation()
        
        overall_scores = []
        phase3_compliant = []
        
        for name, result in validation_results.items():
            overall_scores.append(result.quality_score)
            phase3_compliant.append(result.meets_phase3_requirements())
        
        overall_quality = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        compliance_rate = sum(phase3_compliant) / len(phase3_compliant) if phase3_compliant else 0.0
        
        return {
            'overall_documentation_quality': overall_quality,
            'phase3_compliance_rate': compliance_rate,
            'meets_phase3_threshold': overall_quality >= 0.85,
            'validation_results': validation_results,
            'summary': {
                'total_documents_validated': len(validation_results),
                'compliant_documents': sum(phase3_compliant),
                'quality_score': overall_quality,
                'recommendations': self._generate_workspace_recommendations(validation_results)
            }
        }
    
    def _generate_workspace_recommendations(self, results: Dict[str, DocumentationValidationResult]) -> List[str]:
        """Generate workspace-level documentation recommendations."""
        recommendations = []
        
        low_quality_docs = [name for name, result in results.items() 
                           if result.quality_score < 0.8]
        
        if low_quality_docs:
            recommendations.append(f"Improve quality of: {', '.join(low_quality_docs)}")
        
        total_issues = sum(len(result.issues) for result in results.values())
        if total_issues > 10:
            recommendations.append("Address critical documentation issues across workspace")
        
        docstring_result = results.get('docstring_coverage')
        if docstring_result and docstring_result.metrics.coverage_score < 0.8:
            recommendations.append("Improve docstring coverage in source code")
        
        return recommendations
