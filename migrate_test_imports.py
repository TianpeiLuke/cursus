#!/usr/bin/env python3
"""
Script to remove sys.path boilerplate and update imports in test files.

This script:
1. Removes the repetitive sys.path manipulation code from all test files
2. Updates imports from 'src.cursus.*' to 'cursus.*'
3. Removes unused sys/os imports if they were only used for path setup
4. Creates backups of original files before modification
"""
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple


def backup_file(file_path: Path) -> Path:
    """Create a backup of the original file."""
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    shutil.copy2(file_path, backup_path)
    return backup_path


def remove_boilerplate_code(content: str) -> str:
    """Remove the sys.path boilerplate code from file content."""
    # Pattern to match the boilerplate code block
    boilerplate_patterns = [
        # Main pattern - matches the full boilerplate block
        r'# Add the project root to the Python path.*?sys\.path\.insert\(0, project_root\)\n',
        # Alternative pattern for slight variations
        r'# Add.*?project root.*?Python path.*?sys\.path\.insert\([^)]+\)\n',
        # Pattern for just the sys.path.insert line if comment is different
        r'project_root = os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\)[^)]+\)\)\nif project_root not in sys\.path:\s*\n\s*sys\.path\.insert\(0, project_root\)\n'
    ]
    
    for pattern in boilerplate_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.MULTILINE)
    
    return content


def update_imports(content: str) -> str:
    """Update imports from 'src.cursus.*' to 'cursus.*'."""
    # Update from src.cursus imports
    content = re.sub(r'from src\.cursus\.', 'from cursus.', content)
    content = re.sub(r'import src\.cursus\.', 'import cursus.', content)
    
    return content


def remove_unused_imports(content: str) -> Tuple[str, List[str]]:
    """Remove sys and os imports if they're no longer needed."""
    lines = content.split('\n')
    removed_imports = []
    new_lines = []
    
    for line in lines:
        # Check if this is a sys or os import line
        if re.match(r'^import (sys|os)$', line.strip()):
            # Check if sys/os is used elsewhere in the file (excluding the import line)
            module = line.strip().split()[1]
            rest_of_file = '\n'.join([l for l in lines if l != line])
            
            # Look for usage of the module
            if not re.search(rf'\b{module}\.', rest_of_file):
                removed_imports.append(line.strip())
                continue  # Skip this import line
        
        new_lines.append(line)
    
    return '\n'.join(new_lines), removed_imports


def clean_empty_lines(content: str) -> str:
    """Clean up excessive empty lines that might result from removals."""
    # Replace multiple consecutive empty lines with at most 2
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    return content


def update_test_file(file_path: Path) -> dict:
    """Update a single test file to remove boilerplate and fix imports."""
    result = {
        'file': str(file_path),
        'backup_created': False,
        'boilerplate_removed': False,
        'imports_updated': False,
        'unused_imports_removed': [],
        'error': None
    }
    
    try:
        # Read original content
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Create backup
        backup_path = backup_file(file_path)
        result['backup_created'] = True
        
        # Process content
        content = original_content
        
        # Remove boilerplate code
        new_content = remove_boilerplate_code(content)
        if new_content != content:
            result['boilerplate_removed'] = True
            content = new_content
        
        # Update imports
        new_content = update_imports(content)
        if new_content != content:
            result['imports_updated'] = True
            content = new_content
        
        # Remove unused imports
        new_content, removed_imports = remove_unused_imports(content)
        if removed_imports:
            result['unused_imports_removed'] = removed_imports
            content = new_content
        
        # Clean up empty lines
        content = clean_empty_lines(content)
        
        # Write updated content only if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            # Remove backup if no changes were made
            backup_path.unlink()
            result['backup_created'] = False
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    """Main function to process all test files."""
    test_dir = Path('test')
    
    if not test_dir.exists():
        print(f"Error: Test directory '{test_dir}' not found!")
        return
    
    # Find all test files
    test_files = list(test_dir.rglob('test_*.py'))
    
    if not test_files:
        print("No test files found!")
        return
    
    print(f"Found {len(test_files)} test files to process...")
    print()
    
    results = []
    for test_file in test_files:
        print(f"Processing: {test_file}")
        result = update_test_file(test_file)
        results.append(result)
        
        if result['error']:
            print(f"  ERROR: {result['error']}")
        else:
            changes = []
            if result['boilerplate_removed']:
                changes.append("boilerplate removed")
            if result['imports_updated']:
                changes.append("imports updated")
            if result['unused_imports_removed']:
                changes.append(f"unused imports removed: {', '.join(result['unused_imports_removed'])}")
            
            if changes:
                print(f"  ✓ {', '.join(changes)}")
            else:
                print(f"  - no changes needed")
    
    # Summary
    print()
    print("=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    
    total_files = len(results)
    files_with_changes = sum(1 for r in results if r['boilerplate_removed'] or r['imports_updated'] or r['unused_imports_removed'])
    files_with_errors = sum(1 for r in results if r['error'])
    
    print(f"Total files processed: {total_files}")
    print(f"Files modified: {files_with_changes}")
    print(f"Files with errors: {files_with_errors}")
    
    if files_with_errors > 0:
        print("\nFiles with errors:")
        for result in results:
            if result['error']:
                print(f"  - {result['file']}: {result['error']}")
    
    print(f"\nBackup files created: {sum(1 for r in results if r['backup_created'])}")
    print("Backup files have .backup extension and can be removed after verification.")
    
    print("\nNext steps:")
    print("1. Run tests to verify everything works: pytest test/")
    print("2. If tests pass, remove backup files: find test/ -name '*.backup' -delete")
    print("3. If tests fail, restore from backups and investigate issues")


if __name__ == '__main__':
    main()
