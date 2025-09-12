#!/usr/bin/env python3
"""
Script to fix alignment validation scripts with proper workspace directory definitions.

This script restores the proper workspace directory structure for alignment validation scripts
that were corrupted by the previous project_root fix.
"""

import os
import re
from pathlib import Path
from typing import List


def find_alignment_validation_scripts(base_dir: Path) -> List[Path]:
    """Find all alignment validation scripts that need fixing."""
    validation_dir = base_dir / "test" / "steps" / "scripts" / "alignment_validation"

    if not validation_dir.exists():
        print(f"❌ Validation directory not found: {validation_dir}")
        return []

    scripts = []
    for py_file in validation_dir.glob("validate_*.py"):
        scripts.append(py_file)

    return scripts


def fix_alignment_validation_script(file_path: Path) -> bool:
    """Fix a single alignment validation script."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Check if this file needs fixing (has placeholder strings)
        if '"# Path handled by conftest.py"' not in content:
            return False

        # Replace the corrupted import section
        import_pattern = r"# Add the project root to the Python path\n\)\n\nfrom cursus\.validation\.alignment\.unified_alignment_tester import UnifiedAlignmentTester"

        import_replacement = """# Define workspace directory structure
# workspace_dir points to src/cursus (the main workspace)
current_file = Path(__file__).resolve()
workspace_dir = current_file.parent.parent.parent.parent.parent / "src" / "cursus" / "steps" 

# Define component directories within the workspace
scripts_dir = str(workspace_dir / "scripts")
contracts_dir = str(workspace_dir / "contracts")
specs_dir = str(workspace_dir / "specs")
builders_dir = str(workspace_dir / "builders")
configs_dir = str(workspace_dir / "configs")

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester"""

        content = re.sub(
            import_pattern, import_replacement, content, flags=re.MULTILINE
        )

        # Replace the corrupted UnifiedAlignmentTester initialization
        tester_pattern = r'tester = UnifiedAlignmentTester\(\s*scripts_dir="# Path handled by conftest\.py",\s*contracts_dir="# Path handled by conftest\.py",\s*specs_dir="# Path handled by conftest\.py",\s*builders_dir="# Path handled by conftest\.py"\s*\)'

        tester_replacement = """tester = UnifiedAlignmentTester(
        scripts_dir=scripts_dir,
        contracts_dir=contracts_dir,
        specs_dir=specs_dir,
        builders_dir=builders_dir,
        configs_dir=configs_dir
    )"""

        content = re.sub(
            tester_pattern, tester_replacement, content, flags=re.MULTILINE | re.DOTALL
        )

        # Replace the corrupted script_path in metadata
        metadata_pattern = r"'script_path': \"# Path handled by conftest\.py\""
        metadata_replacement = f"'script_path': str(workspace_dir / 'scripts')"

        content = re.sub(metadata_pattern, metadata_replacement, content)

        # Only write if content changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix all alignment validation scripts."""
    print("🔧 Fixing alignment validation scripts...")

    # Get the project root directory
    script_dir = Path(__file__).parent

    # Find all alignment validation scripts
    print("🔍 Finding alignment validation scripts...")
    scripts = find_alignment_validation_scripts(script_dir)

    if not scripts:
        print("❌ No alignment validation scripts found")
        return

    print(f"📋 Found {len(scripts)} alignment validation scripts")

    # Process each script
    fixed_count = 0

    for script_path in scripts:
        print(f"🔧 Processing {script_path.name}...")

        if fix_alignment_validation_script(script_path):
            fixed_count += 1
            print(f"  ✅ Fixed")
        else:
            print(f"  ⚠️  No changes needed")

    print(f"\n🎉 Processing complete!")
    print(f"📊 Scripts processed: {len(scripts)}")
    print(f"📊 Scripts modified: {fixed_count}")

    print(f"\n💡 Next steps:")
    print(
        f"1. Test with: PYTHONPATH=src python test/steps/scripts/alignment_validation/validate_package.py"
    )
    print(f"2. Verify workspace directory structure is working correctly")


if __name__ == "__main__":
    main()
