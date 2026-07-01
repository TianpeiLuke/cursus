"""
Test Script Analyzer Module

Tests for contract-focused script analysis functionality.
Tests main function signature validation, parameter usage analysis, and contract alignment validation.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from cursus.validation.alignment.analyzer.script_analyzer import ScriptAnalyzer


class TestScriptAnalyzer:
    """Test ScriptAnalyzer contract alignment functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_script(self, content: str) -> str:
        """Create a temporary test script file."""
        script_path = os.path.join(self.temp_dir, "test_script.py")
        with open(script_path, "w") as f:
            f.write(content)
        return script_path

    def test_valid_main_function_signature(self):
        """Test validation of correct main function signature."""
        script_content = '''
import argparse
from typing import Dict

def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
         environ_vars: Dict[str, str], job_args: argparse.Namespace):
    """Main function with correct signature."""
    pass
'''
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        result = analyzer.validate_main_function_signature()

        assert result["has_main"] is True
        assert result["signature_valid"] is True
        assert result["actual_params"] == [
            "input_paths",
            "output_paths",
            "environ_vars",
            "job_args",
        ]
        assert result["expected_params"] == [
            "input_paths",
            "output_paths",
            "environ_vars",
            "job_args",
        ]
        assert len(result["issues"]) == 0

    def test_missing_main_function(self):
        """Test handling of script without main function."""
        script_content = """
def some_other_function():
    pass
"""
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        result = analyzer.validate_main_function_signature()

        assert result["has_main"] is False
        assert result["signature_valid"] is False
        assert "No main function found" in result["issues"]

    def test_incorrect_main_function_signature(self):
        """Test validation of incorrect main function signature."""
        script_content = '''
def main(input_data, output_data):
    """Main function with incorrect signature."""
    pass
'''
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        result = analyzer.validate_main_function_signature()

        assert result["has_main"] is True
        assert result["signature_valid"] is False
        assert result["actual_params"] == ["input_data", "output_data"]
        assert len(result["issues"]) > 0
        # The signature check now requires the 4 required params as a PREFIX (trailing optional
        # params like logger=None are allowed), so the "too few" message is phrased "at least 4".
        assert "Expected at least 4 parameters, got 2" in result["issues"][0]

    def test_parameter_usage_extraction(self):
        """Test extraction of parameter usage patterns."""
        script_content = """
import argparse
from typing import Dict

def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
         environ_vars: Dict[str, str], job_args: argparse.Namespace):
    # Test input_paths usage
    data = input_paths["training_data"]
    validation_data = input_paths.get("validation_data")
    
    # Test output_paths usage
    model_path = output_paths["model"]
    metrics_path = output_paths.get("metrics")
    
    # Test environ_vars usage
    region = environ_vars.get("AWS_REGION")
    bucket = environ_vars.get("S3_BUCKET")
    
    # Test job_args usage
    epochs = job_args.epochs
    learning_rate = job_args.learning_rate
"""
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        usage = analyzer.extract_parameter_usage()

        assert "training_data" in usage["input_paths_keys"]
        assert "validation_data" in usage["input_paths_keys"]
        assert "model" in usage["output_paths_keys"]
        assert "metrics" in usage["output_paths_keys"]
        assert "AWS_REGION" in usage["environ_vars_keys"]
        assert "S3_BUCKET" in usage["environ_vars_keys"]
        assert "epochs" in usage["job_args_attrs"]
        assert "learning_rate" in usage["job_args_attrs"]

    def test_contract_alignment_validation_success(self):
        """Test successful contract alignment validation."""
        script_content = """
import argparse
from typing import Dict

def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
         environ_vars: Dict[str, str], job_args: argparse.Namespace):
    data = input_paths["training_data"]
    model_path = output_paths["model"]
    region = environ_vars.get("AWS_REGION")
    epochs = job_args.epochs
"""
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        contract = {
            "expected_input_paths": {"training_data": {"type": "S3Uri"}},
            "expected_output_paths": {"model": {"type": "S3Uri"}},
            "required_env_vars": ["AWS_REGION"],
            "optional_env_vars": {},
            "expected_arguments": {"epochs": {"type": "int"}},
        }

        issues = analyzer.validate_contract_alignment(contract)

        assert len(issues) == 0

    def test_contract_alignment_validation_errors(self):
        """Test contract alignment validation with errors."""
        script_content = """
import argparse
from typing import Dict

def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
         environ_vars: Dict[str, str], job_args: argparse.Namespace):
    # Using undeclared input path
    data = input_paths["undeclared_input"]
    
    # Using undeclared output path
    model_path = output_paths["undeclared_output"]
    
    # Using undeclared environment variable
    secret = environ_vars.get("UNDECLARED_SECRET")
    
    # Using undeclared job argument
    batch_size = job_args.undeclared_batch_size
"""
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        contract = {
            "expected_input_paths": {},
            "expected_output_paths": {},
            "required_env_vars": [],
            "optional_env_vars": {},
            "expected_arguments": {},
        }

        issues = analyzer.validate_contract_alignment(contract)

        # Should have 4 issues: undeclared input, output, env var, and job arg
        assert len(issues) == 4

        error_categories = [issue["category"] for issue in issues]
        assert "undeclared_input_path" in error_categories
        assert "undeclared_output_path" in error_categories
        assert "undeclared_env_var" in error_categories
        assert "undeclared_job_arg" in error_categories

    def test_script_parsing_error_handling(self):
        """Test handling of script parsing errors."""
        script_content = """
# Invalid Python syntax
def main(input_paths output_paths):
    invalid syntax here
"""
        script_path = self._create_test_script(script_content)

        with pytest.raises(SyntaxError):
            ScriptAnalyzer(script_path)

    def test_empty_script_handling(self):
        """Test handling of empty script."""
        script_content = ""
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        result = analyzer.validate_main_function_signature()
        usage = analyzer.extract_parameter_usage()

        assert result["has_main"] is False
        assert all(len(usage[key]) == 0 for key in usage.keys())

    def test_complex_parameter_usage_patterns(self):
        """Test extraction of complex parameter usage patterns."""
        script_content = """
import argparse
from typing import Dict

def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
         environ_vars: Dict[str, str], job_args: argparse.Namespace):
    # Complex input_paths usage
    for key in ["train", "validation", "test"]:
        if key in input_paths:
            data = input_paths[key]
    
    # Complex output_paths usage
    outputs = {
        "model": output_paths.get("model", "/tmp/model"),
        "metrics": output_paths.get("metrics", "/tmp/metrics")
    }
    
    # Complex environ_vars usage
    config = {
        "region": environ_vars.get("AWS_REGION", "us-west-2"),
        "bucket": environ_vars.get("S3_BUCKET"),
        "role": environ_vars.get("EXECUTION_ROLE")
    }
    
    # Complex job_args usage
    hyperparams = {
        "lr": job_args.learning_rate,
        "epochs": job_args.epochs,
        "batch_size": getattr(job_args, "batch_size", 32)
    }
"""
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        usage = analyzer.extract_parameter_usage()

        # Should detect all the keys used in various patterns
        assert "train" in usage["input_paths_keys"]
        assert "validation" in usage["input_paths_keys"]
        assert "test" in usage["input_paths_keys"]
        assert "model" in usage["output_paths_keys"]
        assert "metrics" in usage["output_paths_keys"]
        assert "AWS_REGION" in usage["environ_vars_keys"]
        assert "S3_BUCKET" in usage["environ_vars_keys"]
        assert "EXECUTION_ROLE" in usage["environ_vars_keys"]
        assert "learning_rate" in usage["job_args_attrs"]
        assert "epochs" in usage["job_args_attrs"]

    def test_real_world_script_patterns(self):
        """Test with patterns similar to real scripts like currency_conversion.py."""
        script_content = '''
import argparse
import os
from typing import Dict

def main(input_paths: Dict[str, str], output_paths: Dict[str, str], 
         environ_vars: Dict[str, str], job_args: argparse.Namespace):
    """Currency conversion script main function."""
    
    # Load input data
    input_data_path = input_paths["input_data"]
    
    # Set up output paths
    output_data_path = output_paths["output_data"]
    
    # Get environment configuration
    aws_region = environ_vars.get("AWS_DEFAULT_REGION", "us-west-2")
    
    # Get job arguments
    conversion_rate = job_args.conversion_rate
    target_currency = job_args.target_currency
    
    # Process data (placeholder)
    print(f"Converting data from {input_data_path} to {target_currency}")
    print(f"Output will be saved to {output_data_path}")
'''
        script_path = self._create_test_script(script_content)
        analyzer = ScriptAnalyzer(script_path)

        # Test signature validation
        signature_result = analyzer.validate_main_function_signature()
        assert signature_result["has_main"] is True
        assert signature_result["signature_valid"] is True

        # Test parameter usage extraction
        usage = analyzer.extract_parameter_usage()
        assert "input_data" in usage["input_paths_keys"]
        assert "output_data" in usage["output_paths_keys"]
        assert "AWS_DEFAULT_REGION" in usage["environ_vars_keys"]
        assert "conversion_rate" in usage["job_args_attrs"]
        assert "target_currency" in usage["job_args_attrs"]

        # Test contract alignment
        contract = {
            "expected_input_paths": {"input_data": {"type": "S3Uri"}},
            "expected_output_paths": {"output_data": {"type": "S3Uri"}},
            "required_env_vars": [],
            "optional_env_vars": {"AWS_DEFAULT_REGION": {"default": "us-west-2"}},
            "expected_arguments": {
                "conversion-rate": {"type": "float"},
                "target-currency": {"type": "str"},
            },
        }

        issues = analyzer.validate_contract_alignment(contract)
        assert len(issues) == 0  # Should align perfectly


class TestReverseAlignment:
    """The reverse-direction checks (FZ 29d14m Cat 4/5): does the script PARSE the args the builder
    passes + READ the required env vars the contract declares?"""

    def _mk(self, src):
        import tempfile, os

        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write(src)
        f.close()
        return ScriptAnalyzer(f.name)

    def test_cat4_missing_argparse_is_error_for_processing(self):
        a = self._mk(
            "import argparse\n"
            "def main(input_paths, output_paths, environ_vars, job_args):\n    return job_args.job_type\n"
            'if __name__ == "__main__":\n    main({}, {}, {}, argparse.Namespace())\n'
        )
        iss = a.validate_reverse_alignment(
            {"job_arguments": [{"flag": "--job_type"}]},
            sagemaker_step_type="Processing",
        )
        cats = [(i["severity"], i["category"]) for i in iss]
        assert ("ERROR", "unparsed_declared_arg") in cats

    def test_cat4_training_vestigial_arg_is_not_error(self):
        """Training/Estimator steps declare job_arguments but get args via JSON not argv -> INFO, not ERROR."""
        a = self._mk(
            "import argparse\n"
            "def main(input_paths, output_paths, environ_vars, job_args):\n    return 1\n"
            'if __name__ == "__main__":\n    main({}, {}, {}, argparse.Namespace())\n'
        )
        iss = a.validate_reverse_alignment(
            {"job_arguments": [{"flag": "--job_type"}]}, sagemaker_step_type="Training"
        )
        assert not any(i["severity"] == "ERROR" for i in iss)
        assert any(
            i["category"] == "reverse_arg_check_skipped_non_processing" for i in iss
        )

    def test_cat4_hyphen_underscore_normalization(self):
        a = self._mk(
            "import argparse\n"
            "def main(input_paths, output_paths, environ_vars, job_args):\n    return 1\n"
            'if __name__ == "__main__":\n    p = argparse.ArgumentParser()\n'
            '    p.add_argument("--job-type")\n    a = p.parse_args()\n    main({}, {}, {}, a)\n'
        )
        iss = a.validate_reverse_alignment(
            {"job_arguments": [{"flag": "--job_type"}]},
            sagemaker_step_type="Processing",
        )
        assert not any(i["category"] == "unparsed_declared_arg" for i in iss)

    def test_cat5_unread_required_env_is_error(self):
        """A required env harvested into a __main__ dict but never consumed in main() is flagged."""
        a = self._mk(
            "import os, argparse\n"
            "def main(input_paths, output_paths, environ_vars, job_args):\n    return 1\n"
            'if __name__ == "__main__":\n'
            '    environ_vars = {"ID_FIELD": os.environ.get("ID_FIELD")}\n'
            "    main({}, {}, environ_vars, None)\n"
        )
        iss = a.validate_reverse_alignment({"env_vars": {"required": ["ID_FIELD"]}})
        assert any(i["category"] == "unread_required_env_var" for i in iss)

    def test_cat5_consumed_env_not_flagged(self):
        a = self._mk(
            "def main(input_paths, output_paths, environ_vars, job_args):\n"
            '    x = environ_vars.get("ID_FIELD")\n    return x\n'
        )
        iss = a.validate_reverse_alignment({"env_vars": {"required": ["ID_FIELD"]}})
        assert not any(i["category"] == "unread_required_env_var" for i in iss)

    def test_optional_env_never_flagged(self):
        a = self._mk(
            "def main(input_paths, output_paths, environ_vars, job_args):\n    return 1\n"
        )
        iss = a.validate_reverse_alignment(
            {"env_vars": {"required": [], "optional": {"FOO": "bar"}}}
        )
        assert not any(i["category"] == "unread_required_env_var" for i in iss)

    def test_optional_logger_param_signature_accepted(self):
        a = self._mk(
            "def main(input_paths, output_paths, environ_vars, job_args, logger=None):\n    return 1\n"
        )
        assert a.validate_main_function_signature()["signature_valid"] is True
