#!/usr/bin/env python3
"""
Script to remove project_root references from all test files.

This script systematically removes project_root path manipulation code
and replaces it with comments indicating that conftest.py handles the setup.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def find_files_with_project_root(test_dir: Path) -> List[Path]:
    """Find all Python files containing project_root references."""
    files_with_project_root = []
    
    for py_file in test_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'project_root' in content:
                    files_with_project_root.append(py_file)
        except Exception as e:
            print(f"⚠️  Could not read {py_file}: {e}")
    
    return files_with_project_root

def fix_project_root_references(file_path: Path) -> bool:
    """Fix project_root references in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Remove project_root definition lines
        patterns_to_remove = [
            r'project_root = os\.path\.abspath\(os\.path\.join\(.*?\)\)',
            r'project_root = Path\(__file__\)\..*?\.parent.*',
            r'project_root = os\.path\.dirname\(.*?\)',
            r'PROJECT_ROOT = Path\(__file__\)\..*?\.parent.*',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '# Note: project_root setup handled by conftest.py', content, flags=re.MULTILINE | re.DOTALL)
        
        # Pattern 2: Remove sys.path.insert lines with project_root
        sys_path_patterns = [
            r'sys\.path\.insert\(0,\s*str\(project_root.*?\)\)',
            r'sys\.path\.insert\(0,\s*project_root.*?\)',
            r'if project_root not in sys\.path:\s*sys\.path\.insert\(0,\s*project_root\)',
            r'if str\(project_root.*?\) not in sys\.path:\s*sys\.path\.insert\(0,\s*str\(project_root.*?\)\)',
        ]
        
        for pattern in sys_path_patterns:
            content = re.sub(pattern, '# Note: sys.path setup is handled by conftest.py', content, flags=re.MULTILINE | re.DOTALL)
        
        # Pattern 3: Replace project_root path constructions with relative paths or comments
        # This is more complex and file-specific, so we'll handle common cases
        
        # Replace project_root / "src" / "cursus" with just comments for now
        content = re.sub(
            r'str\(project_root\s*/\s*"src"\s*/\s*"cursus".*?\)',
            '"# Path handled by conftest.py"',
            content
        )
        
        # Replace other project_root path constructions
        content = re.sub(
            r'project_root\s*/\s*"src".*?',
            '# Path construction handled by conftest.py',
            content
        )
        
        # Pattern 4: Replace os.path.join with project_root
        content = re.sub(
            r'os\.path\.join\(project_root,\s*[\'"]src[\'"],\s*[\'"]cursus[\'"]?\)',
            'os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "cursus")',
            content
        )
        
        content = re.sub(
            r'os\.path\.join\(project_root,\s*[\'"]slipbox[\'"],\s*[\'"]test[\'"]?\)',
            'os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slipbox", "test")',
            content
        )
        
        # Pattern 5: Remove standalone project_root references in path constructions
        # Be careful not to break legitimate uses
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all project_root references."""
    print("🔧 Fixing project_root references in test files...")
    
    # Get the test directory
    script_dir = Path(__file__).parent
    test_dir = script_dir / "test"
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Find all files with project_root references
    print("🔍 Finding files with project_root references...")
    files_with_project_root = find_files_with_project_root(test_dir)
    
    print(f"📋 Found {len(files_with_project_root)} files with project_root references")
    
    # Process each file
    fixed_count = 0
    skipped_files = [
        "conftest.py",  # Keep this one as-is
        "fix_project_root_references.py",  # This script
    ]
    
    for file_path in files_with_project_root:
        if file_path.name in skipped_files:
            print(f"⏭️  Skipping {file_path}")
            continue
            
        print(f"🔧 Processing {file_path.relative_to(script_dir)}...")
        
        if fix_project_root_references(file_path):
            fixed_count += 1
            print(f"  ✅ Fixed")
        else:
            print(f"  ⚠️  No changes needed")
    
    print(f"\n🎉 Processing complete!")
    print(f"📊 Files processed: {len(files_with_project_root)}")
    print(f"📊 Files modified: {fixed_count}")
    print(f"📊 Files skipped: {len(files_with_project_root) - fixed_count}")
    
    print(f"\n💡 Next steps:")
    print(f"1. Review the changes to ensure they're correct")
    print(f"2. Test with: PYTHONPATH=src python -m unittest discover test/")
    print(f"3. Update any remaining import statements to use 'from cursus.*'")

if __name__ == "__main__":
    main()
