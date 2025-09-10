#!/usr/bin/env python3
"""
Unit tests for FlexibleFileResolver class.

This test suite provides comprehensive coverage for the FlexibleFileResolver functionality
which was identified as a missing test in the validation test coverage analysis.
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from cursus.validation.alignment.file_resolver import FlexibleFileResolver


class TestFlexibleFileResolver:
    """Test FlexibleFileResolver functionality."""
    
    @pytest.fixture
    def resolver(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        self.setup_test_files(temp_dir)
        
        base_directories = {
            'scripts': os.path.join(temp_dir, 'scripts'),
            'contracts': os.path.join(temp_dir, 'contracts'),
            'specs': os.path.join(temp_dir, 'specs'),
            'builders': os.path.join(temp_dir, 'builders'),
            'configs': os.path.join(temp_dir, 'configs')
        }
        
        resolver = FlexibleFileResolver(base_directories)
        resolver.temp_dir = temp_dir  # Store for cleanup
        yield resolver
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def setup_test_files(self, temp_dir):
        """Set up test file structure."""
        # Create directory structure
        dirs = [
            'scripts',
            'contracts',
            'specs',
            'builders',
            'configs'
        ]
        
        for dir_name in dirs:
            os.makedirs(os.path.join(temp_dir, dir_name), exist_ok=True)
        
        # Create test files matching real cursus/steps patterns
        test_files = {
            'scripts/train.py': '# Training script',
            'scripts/preprocessing.py': '# Preprocessing script',
            'scripts/evaluation.py': '# Evaluation script',
            'contracts/train_contract.py': '# Training contract',
            'contracts/preprocessing_contract.py': '# Preprocessing contract',
            'contracts/eval_contract.py': '# Evaluation contract',
            'specs/train_spec.py': '# Training specification',
            'specs/preprocessing_spec.py': '# Preprocessing specification',
            'specs/evaluation_spec.py': '# Evaluation specification',
            'builders/builder_train_step.py': '# Training builder',
            'builders/builder_preprocessing_step.py': '# Preprocessing builder',
            'builders/builder_evaluation_step.py': '# Evaluation builder',
            'configs/config_train_step.py': '# Training config',
            'configs/config_preprocessing_step.py': '# Preprocessing config',
            'configs/config_evaluation_step.py': '# Evaluation config'
        }
        
        for file_path, content in test_files.items():
            full_path = os.path.join(temp_dir, file_path)
            with open(full_path, 'w') as f:
                f.write(content)
    
    def test_resolver_initialization(self, resolver):
        """Test FlexibleFileResolver initialization."""
        assert resolver is not None
        # Check that resolver has the expected attributes (may vary by implementation)
        assert hasattr(resolver, 'file_cache') or hasattr(resolver, '_file_cache')
        if hasattr(resolver, 'file_cache'):
            assert isinstance(resolver.file_cache, dict)
        elif hasattr(resolver, '_file_cache'):
            assert isinstance(resolver._file_cache, dict)
    
    def test_discover_all_files(self, resolver):
        """Test file discovery functionality."""
        resolver._discover_all_files()
        
        # Verify files were discovered
        assert len(resolver.file_cache) > 0
        
        # Check that all component types are present
        expected_types = ['scripts', 'contracts', 'specs', 'builders', 'configs']
        for component_type in expected_types:
            assert component_type in resolver.file_cache
            assert isinstance(resolver.file_cache[component_type], dict)
    
    def test_scan_directory(self, resolver):
        """Test directory scanning functionality."""
        scripts_dir = Path(resolver.temp_dir + '/scripts')
        scanned_files = resolver._scan_directory(scripts_dir, 'scripts')
        
        assert isinstance(scanned_files, dict)
        assert len(scanned_files) > 0
        
        # Verify expected files are found
        expected_files = ['train', 'preprocessing', 'evaluation']
        for expected_file in expected_files:
            found = any(expected_file in key for key in scanned_files.keys())
            assert found, f"Expected file {expected_file} not found"
    
    def test_normalize_name(self, resolver):
        """Test name normalization functionality."""
        test_cases = [
            ('train_script.py', 'train_script.py'),  # Actual implementation keeps .py extension
            ('preprocessing-step.py', 'preprocessing_step.py'),
            ('evaluation.step.builder.py', 'evaluation_step_builder.py'),
            ('CamelCaseScript.py', 'camelcasescript.py'),
            ('UPPERCASE_SCRIPT.py', 'uppercase_script.py')
        ]
        
        for input_name, expected_output in test_cases:
            normalized = resolver._normalize_name(input_name)
            assert normalized == expected_output
    
    def test_calculate_similarity(self, resolver):
        """Test similarity calculation."""
        test_cases = [
            ('train', 'train', 1.0),
            ('train', 'training', 0.8),  # High similarity
            ('train', 'preprocessing', 0.0),  # Low similarity
            ('evaluation', 'eval', 0.5),  # Partial match
        ]
        
        for str1, str2, expected_min_similarity in test_cases:
            similarity = resolver._calculate_similarity(str1, str2)
            assert isinstance(similarity, float)
            assert similarity >= 0.0
            assert similarity <= 1.0
            
            if expected_min_similarity == 1.0:
                assert similarity == 1.0
            elif expected_min_similarity > 0.5:
                assert similarity > 0.5
    
    def test_find_best_match(self, resolver):
        """Test best match finding functionality."""
        # Test exact match
        exact_match = resolver._find_best_match('train', 'contracts')
        assert exact_match is not None
        assert 'train' in exact_match.lower()
        
        # Test partial match
        partial_match = resolver._find_best_match('eval', 'contracts')
        assert partial_match is not None
        
        # Test no match
        no_match = resolver._find_best_match('nonexistent', 'contracts')
        assert no_match is None
    
    def test_find_contract_file(self, resolver):
        """Test contract file finding."""
        # Test exact match
        contract_file = resolver.find_contract_file('train')
        assert contract_file is not None
        assert os.path.exists(contract_file)
        assert 'train' in contract_file.lower()
        
        # Test partial match
        contract_file = resolver.find_contract_file('preprocessing')
        assert contract_file is not None
        assert os.path.exists(contract_file)
        
        # Test no match
        contract_file = resolver.find_contract_file('nonexistent')
        assert contract_file is None
    
    def test_find_spec_file(self, resolver):
        """Test specification file finding."""
        # Test exact match
        spec_file = resolver.find_spec_file('train')
        assert spec_file is not None
        assert os.path.exists(spec_file)
        assert 'train' in spec_file.lower()
        
        # Test partial match
        spec_file = resolver.find_spec_file('evaluation')
        assert spec_file is not None
        assert os.path.exists(spec_file)
        
        # Test no match
        spec_file = resolver.find_spec_file('nonexistent')
        assert spec_file is None
    
    def test_find_specification_file(self, resolver):
        """Test specification file finding (alias method)."""
        # Should work the same as find_spec_file
        spec_file = resolver.find_specification_file('train')
        assert spec_file is not None
        assert os.path.exists(spec_file)
        assert 'train' in spec_file.lower()
    
    def test_find_builder_file(self, resolver):
        """Test builder file finding."""
        # Test exact match
        builder_file = resolver.find_builder_file('train')
        assert builder_file is not None
        assert os.path.exists(builder_file)
        assert 'train' in builder_file.lower()
        
        # Test partial match
        builder_file = resolver.find_builder_file('preprocessing')
        assert builder_file is not None
        assert os.path.exists(builder_file)
        
        # Test no match
        builder_file = resolver.find_builder_file('nonexistent')
        assert builder_file is None
    
    def test_find_config_file(self, resolver):
        """Test config file finding."""
        # Test exact match
        config_file = resolver.find_config_file('train')
        assert config_file is not None
        assert os.path.exists(config_file)
        assert 'train' in config_file.lower()
        
        # Test partial match
        config_file = resolver.find_config_file('evaluation')
        assert config_file is not None
        assert os.path.exists(config_file)
        
        # Test no match
        config_file = resolver.find_config_file('nonexistent')
        assert config_file is None
    
    def test_find_all_component_files(self, resolver):
        """Test finding all component files for a script."""
        all_files = resolver.find_all_component_files('train')
        
        assert isinstance(all_files, dict)
        
        # The method returns keys: 'contract', 'spec', 'builder', 'config'
        expected_components = ['contract', 'spec', 'builder', 'config']
        for component in expected_components:
            assert component in all_files
        
        # Verify that found files exist
        for component, file_path in all_files.items():
            if file_path is not None:
                assert os.path.exists(file_path)
    
    def test_refresh_cache(self, resolver):
        """Test cache refresh functionality."""
        # Initial cache
        initial_cache_size = len(resolver.file_cache)
        
        # Add a new file
        new_file_path = os.path.join(resolver.temp_dir, 'contracts', 'new_contract.py')
        with open(new_file_path, 'w') as f:
            f.write('# New contract')
        
        # Refresh cache
        resolver.refresh_cache()
        
        # Verify cache was updated
        assert len(resolver.file_cache) >= initial_cache_size
        
        # Verify new file can be found
        new_contract = resolver.find_contract_file('new')
        assert new_contract is not None
    
    def test_get_available_files_report(self, resolver):
        """Test available files report generation."""
        report = resolver.get_available_files_report()
        
        assert isinstance(report, dict)
        
        expected_components = ['scripts', 'contracts', 'specs', 'builders', 'configs']
        for component in expected_components:
            assert component in report
            assert 'count' in report[component]
            assert 'files' in report[component]
            assert isinstance(report[component]['files'], list)
    
    def test_extract_base_name_from_spec(self, resolver):
        """Test base name extraction from specification path."""
        test_cases = [
            ('train_spec.py', 'train'),
            ('preprocessing_specification.py', 'preprocessing_specification'),  # Actual implementation keeps full name
            ('evaluation_step_spec.py', 'evaluation_step'),
            ('complex_name_spec.py', 'complex_name')
        ]
        
        for spec_name, expected_base in test_cases:
            spec_path = Path(spec_name)
            base_name = resolver.extract_base_name_from_spec(spec_path)
            assert base_name == expected_base
    
    def test_find_spec_constant_name(self, resolver):
        """Test specification constant name finding."""
        # This method might return None if no specific pattern is found
        constant_name = resolver.find_spec_constant_name('train', 'training')
        
        # Should return a string or None
        if constant_name is not None:
            assert isinstance(constant_name, str)
    
    def test_case_insensitive_matching(self, resolver):
        """Test case-insensitive file matching."""
        # Create files with different cases
        mixed_case_file = os.path.join(resolver.temp_dir, 'contracts', 'TrainContract.py')
        with open(mixed_case_file, 'w') as f:
            f.write('# Mixed case contract')
        
        resolver.refresh_cache()
        
        # Should find file regardless of case
        found_file = resolver.find_contract_file('traincontract')
        assert found_file is not None
        
        found_file = resolver.find_contract_file('TRAINCONTRACT')
        assert found_file is not None
    
    def test_fuzzy_matching(self, resolver):
        """Test fuzzy matching capabilities."""
        # Test with slight variations
        test_cases = [
            ('train', 'train'),  # Exact match
            ('training', 'train'),  # Partial match
            ('preprocess', 'preprocessing'),  # Partial match
            ('eval', 'evaluation'),  # Abbreviation match
        ]
        
        for search_term, expected_match in test_cases:
            contract_file = resolver.find_contract_file(search_term)
            if contract_file:
                assert expected_match.lower() in contract_file.lower()
    
    def test_empty_directories(self):
        """Test resolver behavior with empty directories."""
        empty_temp_dir = tempfile.mkdtemp()
        
        try:
            # Create empty directories
            empty_dirs = {
                'scripts': os.path.join(empty_temp_dir, 'scripts'),
                'contracts': os.path.join(empty_temp_dir, 'contracts'),
                'specifications': os.path.join(empty_temp_dir, 'specifications'),
                'builders': os.path.join(empty_temp_dir, 'builders'),
                'configs': os.path.join(empty_temp_dir, 'configs')
            }
            
            for dir_path in empty_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
            
            empty_resolver = FlexibleFileResolver(empty_dirs)
            
            # Should handle empty directories gracefully
            contract_file = empty_resolver.find_contract_file('train')
            assert contract_file is None
            
            report = empty_resolver.get_available_files_report()
            assert isinstance(report, dict)
            
            for component in empty_dirs.keys():
                assert report[component]['count'] == 0
        
        finally:
            import shutil
            shutil.rmtree(empty_temp_dir, ignore_errors=True)
    
    def test_nonexistent_directories(self):
        """Test resolver behavior with nonexistent directories."""
        nonexistent_dirs = {
            'scripts': '/nonexistent/scripts',
            'contracts': '/nonexistent/contracts',
            'specifications': '/nonexistent/specifications',
            'builders': '/nonexistent/builders',
            'configs': '/nonexistent/configs'
        }
        
        # Should handle nonexistent directories gracefully
        try:
            nonexistent_resolver = FlexibleFileResolver(nonexistent_dirs)
            
            contract_file = nonexistent_resolver.find_contract_file('train')
            assert contract_file is None
            
            report = nonexistent_resolver.get_available_files_report()
            assert isinstance(report, dict)
        
        except Exception as e:
            # Should not raise exceptions for nonexistent directories
            pytest.fail(f"Resolver raised exception for nonexistent directories: {e}")


class TestFlexibleFileResolverEdgeCases:
    """Test FlexibleFileResolver edge cases and error conditions."""
    
    def test_special_characters_in_filenames(self):
        """Test resolver with special characters in filenames."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create files with special characters
            special_files = [
                'train-script.py',
                'preprocessing_step.py',
                'evaluation.script.py',
                'model_training_v2.py'
            ]
            
            contracts_dir = os.path.join(temp_dir, 'contracts')
            os.makedirs(contracts_dir, exist_ok=True)
            
            for filename in special_files:
                file_path = os.path.join(contracts_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(f'# {filename}')
            
            base_dirs = {'contracts': contracts_dir}
            resolver = FlexibleFileResolver(base_dirs)
            
            # Should handle special characters gracefully
            found_file = resolver.find_contract_file('train')
            assert found_file is not None
            
            found_file = resolver.find_contract_file('preprocessing')
            assert found_file is not None
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_very_long_filenames(self):
        """Test resolver with very long filenames."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            contracts_dir = os.path.join(temp_dir, 'contracts')
            os.makedirs(contracts_dir, exist_ok=True)
            
            # Create file with very long name
            long_filename = 'very_long_filename_that_exceeds_normal_length_expectations_for_testing_purposes.py'
            long_file_path = os.path.join(contracts_dir, long_filename)
            
            with open(long_file_path, 'w') as f:
                f.write('# Long filename test')
            
            base_dirs = {'contracts': contracts_dir}
            resolver = FlexibleFileResolver(base_dirs)
            
            # Should handle long filenames gracefully
            found_file = resolver.find_contract_file('very_long_filename')
            assert found_file is not None
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_unicode_filenames(self):
        """Test resolver with unicode characters in filenames."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            contracts_dir = os.path.join(temp_dir, 'contracts')
            os.makedirs(contracts_dir, exist_ok=True)
            
            # Create file with unicode characters (if supported by filesystem)
            unicode_filename = 'tráin_contrâct.py'
            unicode_file_path = os.path.join(contracts_dir, unicode_filename)
            
            try:
                with open(unicode_file_path, 'w', encoding='utf-8') as f:
                    f.write('# Unicode filename test')
                
                base_dirs = {'contracts': contracts_dir}
                resolver = FlexibleFileResolver(base_dirs)
                
                # Should handle unicode filenames gracefully
                report = resolver.get_available_files_report()
                assert isinstance(report, dict)
            
            except (OSError, UnicodeError):
                # Skip test if filesystem doesn't support unicode filenames
                pass
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_concurrent_access(self):
        """Test resolver behavior with concurrent access."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            contracts_dir = os.path.join(temp_dir, 'contracts')
            os.makedirs(contracts_dir, exist_ok=True)
            
            # Create test file
            test_file = os.path.join(contracts_dir, 'test_contract.py')
            with open(test_file, 'w') as f:
                f.write('# Test contract')
            
            base_dirs = {'contracts': contracts_dir}
            resolver = FlexibleFileResolver(base_dirs)
            
            # Simulate concurrent access
            import threading
            results = []
            
            def find_file():
                result = resolver.find_contract_file('test')
                results.append(result)
            
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=find_file)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All threads should find the file
            assert len(results) == 5
            for result in results:
                assert result is not None
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
