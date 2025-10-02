"""
Import analyzer for analyzing import statements and framework usage in scripts.

Provides analysis of module imports, framework requirements, and dependency
patterns to support alignment validation with StepCatalog integration.
"""

import re
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict

from ..utils.script_analysis_models import ImportStatement


class ImportAnalyzer:
    """
    Analyzes import statements and framework usage in Python scripts.

    Uses StepCatalog integration for:
    - Built-in framework detection
    - Metadata-driven analysis
    - Workspace-aware discovery
    - Registry-based insights
    """

    def __init__(self, imports: List[ImportStatement] = None, script_content: str = None, step_catalog=None):
        """
        Initialize the import analyzer with optional StepCatalog integration.

        Args:
            imports: List of import statements extracted from the script (legacy)
            script_content: Full content of the script for usage analysis (legacy)
            step_catalog: Optional StepCatalog instance for enhanced analysis
        """
        self.imports = imports or []
        self.script_content = script_content or ""
        self.step_catalog = step_catalog

    def analyze_step_imports(self, step_name: str) -> Dict[str, Any]:
        """
        Analyze imports for a step using StepCatalog integration (preferred method).

        Args:
            step_name: Name of the step to analyze

        Returns:
            Dictionary containing comprehensive import analysis
        """
        if not self.step_catalog:
            return {"error": "StepCatalog not available for step analysis"}

        try:
            # Get step information from StepCatalog
            step_info = self.step_catalog.get_step_info(step_name)
            if not step_info:
                return {"error": f"Step {step_name} not found in StepCatalog"}

            # Get script file path from metadata
            script_metadata = step_info.file_components.get('script')
            if not script_metadata:
                return {"error": f"Script file metadata not found for step {step_name}"}

            # Load script content
            if not script_metadata.path.exists():
                return {"error": f"Script file not accessible: {script_metadata.path}"}

            with open(script_metadata.path, 'r', encoding='utf-8') as f:
                script_content = f.read()

            # Extract imports from script content
            imports = self._extract_imports_from_content(script_content)

            # Create temporary analyzer instance for legacy methods
            temp_analyzer = ImportAnalyzer(imports, script_content)

            # Perform comprehensive analysis
            analysis = {
                "step_name": step_info.step_name,
                "workspace_id": step_info.workspace_id,
                "registry_data": step_info.registry_data,
                "file_path": str(script_metadata.path),
                "last_modified": script_metadata.modified_time.isoformat() if script_metadata.modified_time else None,
                
                # Use StepCatalog's built-in framework detection
                "framework": self.step_catalog.detect_framework(step_name),
                
                # Enhanced analysis using legacy methods with StepCatalog context
                "import_categories": temp_analyzer.categorize_imports(),
                "framework_requirements": temp_analyzer.extract_framework_requirements(),
                "import_usage": temp_analyzer.analyze_import_usage(),
                "unused_imports": len(temp_analyzer.find_unused_imports()),
                "import_organization": temp_analyzer.check_import_organization(),
                "import_summary": temp_analyzer.get_import_summary(),
                
                # StepCatalog-enhanced metadata
                "total_imports": len(imports),
                "has_framework_imports": bool(temp_analyzer.extract_framework_requirements()),
            }

            return analysis

        except Exception as e:
            return {
                "error": str(e),
                "step_name": step_name,
                "import_categories": {"standard_library": [], "third_party": [], "local": [], "unknown": []},
                "framework_requirements": {},
                "total_imports": 0,
            }

    def _extract_imports_from_content(self, script_content: str) -> List[ImportStatement]:
        """
        Extract import statements from script content.

        Args:
            script_content: Content of the script

        Returns:
            List of ImportStatement objects
        """
        imports = []
        lines = script_content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Handle 'import module' statements
            if line.startswith('import '):
                import_part = line[7:].strip()
                if ' as ' in import_part:
                    module_name, alias = import_part.split(' as ', 1)
                    module_name = module_name.strip()
                    alias = alias.strip()
                else:
                    module_name = import_part.strip()
                    alias = None
                
                imports.append(ImportStatement(
                    module_name=module_name,
                    import_alias=alias,
                    is_from_import=False,
                    imported_items=[],
                    line_number=line_num
                ))
            
            # Handle 'from module import items' statements
            elif line.startswith('from ') and ' import ' in line:
                parts = line.split(' import ', 1)
                module_name = parts[0][5:].strip()  # Remove 'from '
                import_items = [item.strip() for item in parts[1].split(',')]
                
                imports.append(ImportStatement(
                    module_name=module_name,
                    import_alias=None,
                    is_from_import=True,
                    imported_items=import_items,
                    line_number=line_num
                ))
        
        return imports

    def categorize_imports(self) -> Dict[str, List[ImportStatement]]:
        """Categorize imports by type (standard library, third-party, local)."""
        categories = {
            "standard_library": [],
            "third_party": [],
            "local": [],
            "unknown": [],
        }

        # Standard library modules (common ones)
        standard_modules = {
            "os",
            "sys",
            "json",
            "csv",
            "logging",
            "argparse",
            "datetime",
            "pathlib",
            "re",
            "collections",
            "itertools",
            "functools",
            "typing",
            "dataclasses",
            "enum",
            "abc",
            "copy",
            "pickle",
            "urllib",
            "http",
            "email",
            "xml",
            "html",
            "sqlite3",
            "threading",
            "multiprocessing",
            "subprocess",
            "shutil",
            "tempfile",
            "glob",
            "fnmatch",
            "math",
            "random",
            "statistics",
            "decimal",
            "fractions",
            "hashlib",
            "hmac",
            "secrets",
            "time",
            "calendar",
            "zoneinfo",
            "locale",
            "gettext",
            "io",
            "codecs",
            "unicodedata",
            "stringprep",
            "readline",
            "rlcompleter",
            "struct",
            "array",
            "weakref",
            "types",
            "gc",
            "inspect",
            "site",
            "importlib",
            "pkgutil",
            "modulefinder",
            "runpy",
            "ast",
            "symtable",
            "symbol",
            "token",
            "keyword",
            "tokenize",
            "tabnanny",
            "pyclbr",
            "py_compile",
            "compileall",
            "dis",
            "pickletools",
            "platform",
            "errno",
            "ctypes",
            "mmap",
            "winreg",
            "msvcrt",
            "winsound",
            "posix",
            "pwd",
            "spwd",
            "grp",
            "crypt",
            "termios",
            "tty",
            "pty",
            "fcntl",
            "pipes",
            "resource",
            "nis",
            "syslog",
            "optparse",
            "getopt",
        }

        # Common third-party data science and ML modules
        third_party_modules = {
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "seaborn",
            "sklearn",
            "scikit-learn",
            "tensorflow",
            "torch",
            "pytorch",
            "keras",
            "xgboost",
            "lightgbm",
            "catboost",
            "joblib",
            "requests",
            "urllib3",
            "boto3",
            "botocore",
            "awscli",
            "sagemaker",
            "pydantic",
            "fastapi",
            "flask",
            "django",
            "sqlalchemy",
            "psycopg2",
            "pymongo",
            "redis",
            "celery",
            "pytest",
            "unittest2",
            "mock",
            "coverage",
            "tox",
            "click",
            "typer",
            "rich",
            "colorama",
            "tqdm",
            "pillow",
            "opencv",
            "imageio",
            "plotly",
            "bokeh",
            "jupyter",
            "ipython",
            "notebook",
            "jupyterlab",
        }

        for import_stmt in self.imports:
            module_name = import_stmt.module_name
            root_module = module_name.split(".")[0]

            if root_module in standard_modules:
                categories["standard_library"].append(import_stmt)
            elif root_module in third_party_modules:
                categories["third_party"].append(import_stmt)
            elif module_name.startswith(".") or root_module in ["src", "cursus"]:
                categories["local"].append(import_stmt)
            else:
                categories["unknown"].append(import_stmt)

        return categories

    def extract_framework_requirements(self) -> Dict[str, Optional[str]]:
        """Extract framework requirements and their potential versions."""
        requirements = {}

        # Framework mapping with common import patterns
        framework_patterns = {
            "pandas": ["pandas", "pd"],
            "numpy": ["numpy", "np"],
            "scikit-learn": ["sklearn", "scikit-learn"],
            "tensorflow": ["tensorflow", "tf"],
            "torch": ["torch", "pytorch"],
            "xgboost": ["xgboost", "xgb"],
            "lightgbm": ["lightgbm", "lgb"],
            "matplotlib": ["matplotlib", "matplotlib.pyplot"],
            "seaborn": ["seaborn", "sns"],
            "boto3": ["boto3"],
            "sagemaker": ["sagemaker"],
            "requests": ["requests"],
            "pydantic": ["pydantic"],
            "joblib": ["joblib"],
        }

        # Check which frameworks are imported
        imported_modules = {imp.module_name for imp in self.imports}
        imported_aliases = {
            imp.import_alias for imp in self.imports if imp.import_alias
        }

        for framework, patterns in framework_patterns.items():
            for pattern in patterns:
                if (
                    pattern in imported_modules
                    or pattern in imported_aliases
                    or any(pattern in module for module in imported_modules)
                ):

                    # Try to extract version from comments or requirements
                    version = self._extract_version_hint(framework)
                    requirements[framework] = version
                    break

        return requirements

    def analyze_import_usage(self) -> Dict[str, Dict[str, Any]]:
        """Analyze how imported modules are used in the script."""
        usage_analysis = {}

        for import_stmt in self.imports:
            module_name = import_stmt.module_name
            alias = import_stmt.import_alias or module_name

            # Count usage occurrences
            if import_stmt.is_from_import:
                # For 'from X import Y', count usage of imported items
                usage_count = 0
                for item in import_stmt.imported_items:
                    usage_count += len(
                        re.findall(rf"\b{re.escape(item)}\b", self.script_content)
                    )
            else:
                # For 'import X' or 'import X as Y', count usage of module/alias
                usage_count = len(
                    re.findall(rf"\b{re.escape(alias)}\b", self.script_content)
                )
                # Subtract the import statement itself
                usage_count = max(0, usage_count - 1)

            # Analyze usage patterns
            usage_patterns = self._analyze_usage_patterns(alias, import_stmt)

            usage_analysis[module_name] = {
                "import_type": (
                    "from_import" if import_stmt.is_from_import else "direct_import"
                ),
                "alias": alias,
                "usage_count": usage_count,
                "is_used": usage_count > 0,
                "usage_patterns": usage_patterns,
                "line_number": import_stmt.line_number,
            }

        return usage_analysis

    def find_unused_imports(self) -> List[ImportStatement]:
        """Find imports that are not used in the script."""
        usage_analysis = self.analyze_import_usage()
        unused_imports = []

        for import_stmt in self.imports:
            module_name = import_stmt.module_name
            if (
                module_name in usage_analysis
                and not usage_analysis[module_name]["is_used"]
            ):
                unused_imports.append(import_stmt)

        return unused_imports

    def find_missing_imports(self, required_modules: List[str]) -> List[str]:
        """Find required modules that are not imported."""
        imported_modules = {imp.module_name.split(".")[0] for imp in self.imports}
        imported_aliases = {
            imp.import_alias for imp in self.imports if imp.import_alias
        }

        # Also check for from imports
        from_imports = set()
        for imp in self.imports:
            if imp.is_from_import:
                from_imports.update(imp.imported_items)

        all_imported = imported_modules | imported_aliases | from_imports

        missing = []
        for required in required_modules:
            if required not in all_imported:
                # Check if it might be imported under a different name
                if not any(required in imp.module_name for imp in self.imports):
                    missing.append(required)

        return missing

    def check_import_organization(self) -> Dict[str, Any]:
        """Check import organization and style compliance."""
        issues = []
        suggestions = []

        # Group imports by line numbers to check organization
        import_lines = [(imp.line_number, imp) for imp in self.imports]
        import_lines.sort()

        # Check for PEP 8 import organization
        categories = self.categorize_imports()

        # Expected order: standard library, third-party, local
        expected_order = ["standard_library", "third_party", "local"]

        current_category = None
        last_line = 0

        for line_num, import_stmt in import_lines:
            # Determine category of this import
            import_category = None
            for category, imports in categories.items():
                if import_stmt in imports:
                    import_category = category
                    break

            if import_category and import_category != "unknown":
                if current_category is None:
                    current_category = import_category
                elif import_category != current_category:
                    # Check if this is a valid transition
                    try:
                        current_idx = expected_order.index(current_category)
                        new_idx = expected_order.index(import_category)

                        if new_idx < current_idx:
                            issues.append(
                                {
                                    "type": "import_order",
                                    "line": line_num,
                                    "message": f"{import_category} import after {current_category} import",
                                    "suggestion": "Organize imports: standard library, third-party, local",
                                }
                            )
                    except ValueError:
                        pass  # Category not in expected order

                    current_category = import_category

            # Check for gaps between imports (should be grouped)
            if last_line > 0 and line_num - last_line > 2:
                suggestions.append(
                    {
                        "type": "import_spacing",
                        "line": line_num,
                        "message": "Large gap between imports",
                        "suggestion": "Group related imports together",
                    }
                )

            last_line = line_num

        return {
            "issues": issues,
            "suggestions": suggestions,
            "total_imports": len(self.imports),
            "categories": {k: len(v) for k, v in categories.items()},
        }

    def get_import_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of import analysis."""
        categories = self.categorize_imports()
        requirements = self.extract_framework_requirements()
        usage_analysis = self.analyze_import_usage()
        unused_imports = self.find_unused_imports()
        organization = self.check_import_organization()

        return {
            "total_imports": len(self.imports),
            "categories": {k: len(v) for k, v in categories.items()},
            "framework_requirements": requirements,
            "unused_imports": len(unused_imports),
            "usage_summary": {
                "used_imports": sum(
                    1 for analysis in usage_analysis.values() if analysis["is_used"]
                ),
                "unused_imports": sum(
                    1 for analysis in usage_analysis.values() if not analysis["is_used"]
                ),
            },
            "organization_issues": len(organization["issues"]),
            "organization_suggestions": len(organization["suggestions"]),
        }

    def _extract_version_hint(self, framework: str) -> Optional[str]:
        """Try to extract version hints from comments or docstrings."""
        # Look for version patterns in comments
        version_patterns = [
            rf"{framework}\s*==\s*([0-9.]+)",
            rf"{framework}\s*>=\s*([0-9.]+)",
            rf"{framework}\s*~=\s*([0-9.]+)",
            rf"{framework}\s*version\s*([0-9.]+)",
            rf"{framework}\s*v([0-9.]+)",
        ]

        for pattern in version_patterns:
            matches = re.findall(pattern, self.script_content, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _analyze_usage_patterns(
        self, alias: str, import_stmt: ImportStatement
    ) -> List[str]:
        """Analyze how a module is used in the script."""
        patterns = []

        # Look for common usage patterns
        if import_stmt.is_from_import:
            for item in import_stmt.imported_items:
                # Function calls
                if re.search(rf"\b{re.escape(item)}\s*\(", self.script_content):
                    patterns.append("function_call")

                # Class instantiation
                if re.search(rf"\b{re.escape(item)}\s*\(.*\)", self.script_content):
                    patterns.append("class_instantiation")

                # Attribute access
                if re.search(rf"\b{re.escape(item)}\.[a-zA-Z_]", self.script_content):
                    patterns.append("attribute_access")
        else:
            # Module usage patterns
            if re.search(rf"\b{re.escape(alias)}\.[a-zA-Z_]", self.script_content):
                patterns.append("attribute_access")

            if re.search(rf"\b{re.escape(alias)}\s*\(", self.script_content):
                patterns.append("direct_call")

        return list(set(patterns))  # Remove duplicates
