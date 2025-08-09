"""
Test suite for PathReference model.
"""

import unittest

from src.cursus.validation.alignment.alignment_utils import PathReference


class TestPathReference(unittest.TestCase):
    """Test PathReference model."""
    
    def test_path_reference_creation(self):
        """Test basic PathReference creation."""
        path_ref = PathReference(
            path="/opt/ml/input/data",
            line_number=10,
            context="with open('/opt/ml/input/data/file.csv', 'r') as f:",
            is_hardcoded=True
        )
        
        self.assertEqual(path_ref.path, "/opt/ml/input/data")
        self.assertEqual(path_ref.line_number, 10)
        self.assertEqual(path_ref.context, "with open('/opt/ml/input/data/file.csv', 'r') as f:")
        self.assertTrue(path_ref.is_hardcoded)
        self.assertIsNone(path_ref.construction_method)
    
    def test_path_reference_with_construction_method(self):
        """Test PathReference creation with construction method."""
        path_ref = PathReference(
            path="os.path.join(base_dir, 'data')",
            line_number=25,
            context="data_path = os.path.join(base_dir, 'data')",
            is_hardcoded=False,
            construction_method="os.path.join"
        )
        
        self.assertEqual(path_ref.construction_method, "os.path.join")
        self.assertFalse(path_ref.is_hardcoded)
    
    def test_path_reference_dynamic_path(self):
        """Test PathReference for dynamic path construction."""
        path_ref = PathReference(
            path="os.environ.get('SM_CHANNEL_TRAINING')",
            line_number=15,
            context="training_path = os.environ.get('SM_CHANNEL_TRAINING')",
            is_hardcoded=False,
            construction_method="environment_variable"
        )
        
        self.assertFalse(path_ref.is_hardcoded)
        self.assertEqual(path_ref.construction_method, "environment_variable")
    
    def test_path_reference_defaults(self):
        """Test PathReference default values."""
        path_ref = PathReference(
            path="/some/path",
            line_number=5,
            context="path = '/some/path'"
        )
        
        self.assertTrue(path_ref.is_hardcoded)  # Default is True
        self.assertIsNone(path_ref.construction_method)
    
    def test_path_reference_string_representation(self):
        """Test PathReference string representation."""
        path_ref = PathReference(
            path="/test/path",
            line_number=20,
            context="test_path = '/test/path'",
            is_hardcoded=True
        )
        
        str_repr = str(path_ref)
        self.assertIn("/test/path", str_repr)
        self.assertIn("20", str_repr)
    
    def test_path_reference_serialization(self):
        """Test PathReference serialization to dict."""
        path_ref = PathReference(
            path="/opt/ml/processing/input/data.csv",
            line_number=42,
            context="data = pd.read_csv('/opt/ml/processing/input/data.csv')",
            is_hardcoded=True,
            construction_method="literal"
        )
        
        path_dict = path_ref.dict()
        
        self.assertEqual(path_dict["path"], "/opt/ml/processing/input/data.csv")
        self.assertEqual(path_dict["line_number"], 42)
        self.assertEqual(path_dict["context"], "data = pd.read_csv('/opt/ml/processing/input/data.csv')")
        self.assertTrue(path_dict["is_hardcoded"])
        self.assertEqual(path_dict["construction_method"], "literal")
    
    def test_path_reference_validation(self):
        """Test PathReference validation with invalid data."""
        # Test with missing required fields
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PathReference(
                path="/test/path"
                # Missing required line_number and context
            )


if __name__ == '__main__':
    unittest.main()
