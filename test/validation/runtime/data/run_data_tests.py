#!/usr/bin/env python3
"""Test runner for data management components."""

import subprocess
import sys
import os
from pathlib import Path

def run_data_tests():
    """Run all data management tests using unittest discovery."""
    # Get the project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent.parent
    
    # Change to project root directory to ensure proper imports
    original_cwd = os.getcwd()
    os.chdir(str(project_root))
    
    try:
        # Use subprocess to run unittest discovery (the approach that works)
        cmd = [
            sys.executable, "-m", "unittest", "discover",
            "-s", "test/validation/runtime/data",
            "-p", "test_*.py",
            "-v"
        ]
        
        print("Running data management tests...")
        print("Command:", " ".join(cmd))
        print("Working directory:", str(project_root))
        print("="*70)
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Parse the output to extract test statistics using robust multi-layer approach
        # Combine both stdout and stderr as test output can appear in either
        combined_output = ""
        if result.stdout:
            combined_output += result.stdout
        if result.stderr:
            combined_output += "\n" + result.stderr
        
        output_lines = combined_output.split('\n') if combined_output else []
        full_output = combined_output
        
        tests_run = 0
        failures = 0
        errors = 0
        success = False
        
        # Layer 1: Regex-based primary parsing
        import re
        
        # Pattern for "Ran X tests in Y.Zs" with flexible whitespace
        ran_pattern = r'Ran\s+(\d+)\s+tests?\s+in\s+[\d.]+s'
        ran_match = re.search(ran_pattern, full_output, re.IGNORECASE)
        if ran_match:
            tests_run = int(ran_match.group(1))
        
        # Pattern for failure/error counts: "FAILED (failures=X, errors=Y)"
        failed_pattern = r'FAILED\s*\((?:.*?failures=(\d+))?(?:.*?errors=(\d+))?\)'
        failed_match = re.search(failed_pattern, full_output, re.IGNORECASE)
        if failed_match:
            if failed_match.group(1):
                failures = int(failed_match.group(1))
            if failed_match.group(2):
                errors = int(failed_match.group(2))
        
        # Check for success indicator
        if re.search(r'\bOK\b', full_output):
            success = True
        
        # Layer 2: Line-by-line parsing as backup
        if tests_run == 0:
            for line in output_lines:
                line_stripped = line.strip()
                
                # Look for "Ran X tests" pattern with more flexibility
                if "ran" in line_stripped.lower() and "test" in line_stripped.lower():
                    # Extract all numbers from the line
                    numbers = re.findall(r'\d+', line_stripped)
                    if numbers:
                        # Take the first number as test count
                        tests_run = int(numbers[0])
                        break
                
                # Alternative patterns
                if line_stripped.startswith("Ran ") or " Ran " in line_stripped:
                    parts = line_stripped.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "ran" and i + 1 < len(parts):
                            try:
                                tests_run = int(parts[i + 1])
                                break
                            except ValueError:
                                continue
        
        # Layer 3: Count test method executions as fallback
        if tests_run == 0:
            # Count lines that look like test executions
            test_execution_patterns = [
                r'test_\w+.*?\.\.\..*?ok',
                r'test_\w+.*?\.\.\..*?FAIL',
                r'test_\w+.*?\.\.\..*?ERROR',
                r'test_\w+\s*\(.*?\)\s*\.\.\..*?(?:ok|FAIL|ERROR)',
            ]
            
            test_count = 0
            for pattern in test_execution_patterns:
                matches = re.findall(pattern, full_output, re.IGNORECASE | re.MULTILINE)
                test_count += len(matches)
            
            if test_count > 0:
                tests_run = test_count
        
        # Layer 4: Enhanced failure/error parsing
        if failures == 0 and errors == 0:
            # Count FAIL and ERROR occurrences in test output
            fail_count = len(re.findall(r'\.\.\..*?FAIL', full_output, re.IGNORECASE))
            error_count = len(re.findall(r'\.\.\..*?ERROR', full_output, re.IGNORECASE))
            
            if fail_count > 0:
                failures = fail_count
            if error_count > 0:
                errors = error_count
        
        # Layer 5: Validation and correction
        # If we found tests but no success/failure indicator, infer from return code
        if tests_run > 0 and not success and failures == 0 and errors == 0:
            if result.returncode == 0:
                success = True
        
        # If we have failures/errors but no test count, estimate
        if tests_run == 0 and (failures > 0 or errors > 0):
            tests_run = failures + errors  # Minimum estimate
        
        # Debug output for troubleshooting
        if tests_run == 0:
            print("\nDEBUG: Parsing failed, raw output analysis:")
            print(f"Output length: {len(full_output)} characters")
            print(f"Return code: {result.returncode}")
            print("First 500 characters of output:")
            print(repr(full_output[:500]))
            print("Last 500 characters of output:")
            print(repr(full_output[-500:]))
        
        # Print summary
        print("\n" + "="*70)
        print("DATA MANAGEMENT MODULE TEST SUMMARY")
        print("="*70)
        print(f"Tests run: {tests_run}")
        print(f"Failures: {failures}")
        print(f"Errors: {errors}")
        print(f"Skipped: 0")  # unittest discover doesn't easily report skipped
        
        success_rate = ((tests_run - failures - errors) / tests_run * 100) if tests_run > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        return result.returncode == 0
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == '__main__':
    success = run_data_tests()
    sys.exit(0 if success else 1)
