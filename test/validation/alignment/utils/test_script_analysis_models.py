"""
Test suite for script analysis models.
"""

import unittest

from cursus.validation.alignment.alignment_utils import (
    PathReference,
    EnvVarAccess,
    ImportStatement,
    ArgumentDefinition,
    PathConstruction,
    FileOperation
)

class TestPathReference(unittest.TestCase):
    """Test PathReference model."""
    
    def test_path_reference_creation(self):
        """Test basic PathReference creation."""
        path_ref = PathReference(
            path="/opt/ml/input/data/train",
            line_number=10,
            context="data loading"
        )
        
        self.assertEqual(path_ref.path, "/opt/ml/input/data/train")
        self.assertEqual(path_ref.line_number, 10)
        self.assertEqual(path_ref.context, "data loading")
    
    def test_path_reference_defaults(self):
        """Test PathReference default values."""
        path_ref = PathReference(
            path="/opt/ml/model",
            line_number=1,
            context="model loading"
        )
        
        self.assertEqual(path_ref.path, "/opt/ml/model")
        self.assertEqual(path_ref.line_number, 1)
        self.assertEqual(path_ref.context, "model loading")
        self.assertTrue(path_ref.is_hardcoded)  # Default value
        self.assertIsNone(path_ref.construction_method)  # Default value
    
    def test_path_reference_serialization(self):
        """Test PathReference serialization."""
        path_ref = PathReference(
            path="/opt/ml/processing/output",
            line_number=25,
            context="output saving"
        )
        
        path_dict = path_ref.model_dump()
        self.assertEqual(path_dict["path"], "/opt/ml/processing/output")
        self.assertEqual(path_dict["line_number"], 25)
        self.assertEqual(path_dict["context"], "output saving")

class TestEnvVarAccess(unittest.TestCase):
    """Test EnvVarAccess model."""
    
    def test_env_var_access_creation(self):
        """Test basic EnvVarAccess creation."""
        env_var = EnvVarAccess(
            variable_name="SM_CHANNEL_TRAIN",
            line_number=15,
            context="data loading",
            access_method="os.environ.get"
        )
        
        self.assertEqual(env_var.variable_name, "SM_CHANNEL_TRAIN")
        self.assertEqual(env_var.line_number, 15)
        self.assertEqual(env_var.access_method, "os.environ.get")
    
    def test_env_var_access_defaults(self):
        """Test EnvVarAccess default values."""
        env_var = EnvVarAccess(
            variable_name="SM_MODEL_DIR",
            line_number=10,
            context="model directory access",
            access_method="os.environ"
        )
        
        self.assertEqual(env_var.variable_name, "SM_MODEL_DIR")
        self.assertEqual(env_var.line_number, 10)
        self.assertEqual(env_var.access_method, "os.environ")
        self.assertFalse(env_var.has_default)  # Default value
        self.assertIsNone(env_var.default_value)  # Default value
    
    def test_env_var_access_serialization(self):
        """Test EnvVarAccess serialization."""
        env_var = EnvVarAccess(
            variable_name="SM_CHANNEL_VALIDATION",
            line_number=20,
            context="validation data access",
            access_method="os.getenv"
        )
        
        env_dict = env_var.model_dump()
        self.assertEqual(env_dict["variable_name"], "SM_CHANNEL_VALIDATION")
        self.assertEqual(env_dict["line_number"], 20)
        self.assertEqual(env_dict["access_method"], "os.getenv")

class TestImportStatement(unittest.TestCase):
    """Test ImportStatement model."""
    
    def test_import_statement_creation(self):
        """Test basic ImportStatement creation."""
        import_stmt = ImportStatement(
            module_name="xgboost",
            import_alias="xgb",
            line_number=1
        )
        
        self.assertEqual(import_stmt.module_name, "xgboost")
        self.assertEqual(import_stmt.import_alias, "xgb")
        self.assertEqual(import_stmt.line_number, 1)
    
    def test_import_statement_no_alias(self):
        """Test ImportStatement without alias."""
        import_stmt = ImportStatement(
            module_name="pandas",
            import_alias=None,
            line_number=2
        )
        
        self.assertEqual(import_stmt.module_name, "pandas")
        self.assertIsNone(import_stmt.import_alias)
        self.assertEqual(import_stmt.line_number, 2)
    
    def test_import_statement_defaults(self):
        """Test ImportStatement default values."""
        import_stmt = ImportStatement(
            module_name="torch",
            import_alias=None,
            line_number=1
        )
        
        self.assertEqual(import_stmt.module_name, "torch")
        self.assertIsNone(import_stmt.import_alias)
        self.assertEqual(import_stmt.line_number, 1)
        self.assertFalse(import_stmt.is_from_import)  # Default value
        self.assertEqual(import_stmt.imported_items, [])  # Default value
    
    def test_import_statement_serialization(self):
        """Test ImportStatement serialization."""
        import_stmt = ImportStatement(
            module_name="sklearn.ensemble",
            import_alias=None,
            line_number=5
        )
        
        import_dict = import_stmt.model_dump()
        self.assertEqual(import_dict["module_name"], "sklearn.ensemble")
        self.assertIsNone(import_dict["import_alias"])
        self.assertEqual(import_dict["line_number"], 5)

class TestArgumentDefinition(unittest.TestCase):
    """Test ArgumentDefinition model."""
    
    def test_argument_definition_creation(self):
        """Test basic ArgumentDefinition creation."""
        arg_def = ArgumentDefinition(
            argument_name="train_data_path",
            line_number=10,
            argument_type="str",
            default_value="/opt/ml/input/data/train"
        )
        
        self.assertEqual(arg_def.argument_name, "train_data_path")
        self.assertEqual(arg_def.argument_type, "str")
        self.assertEqual(arg_def.default_value, "/opt/ml/input/data/train")
        self.assertEqual(arg_def.line_number, 10)
    
    def test_argument_definition_no_default(self):
        """Test ArgumentDefinition without default value."""
        arg_def = ArgumentDefinition(
            argument_name="model_name",
            line_number=12,
            argument_type="str"
        )
        
        self.assertEqual(arg_def.argument_name, "model_name")
        self.assertEqual(arg_def.argument_type, "str")
        self.assertIsNone(arg_def.default_value)
        self.assertEqual(arg_def.line_number, 12)
    
    def test_argument_definition_defaults(self):
        """Test ArgumentDefinition default values."""
        arg_def = ArgumentDefinition(
            argument_name="learning_rate",
            line_number=5
        )
        
        self.assertEqual(arg_def.argument_name, "learning_rate")
        self.assertIsNone(arg_def.argument_type)
        self.assertIsNone(arg_def.default_value)
        self.assertEqual(arg_def.line_number, 5)
        self.assertFalse(arg_def.is_required)  # Default value
        self.assertFalse(arg_def.has_default)  # Default value
    
    def test_argument_definition_serialization(self):
        """Test ArgumentDefinition serialization."""
        arg_def = ArgumentDefinition(
            argument_name="epochs",
            line_number=8,
            argument_type="int",
            default_value="100"
        )
        
        arg_dict = arg_def.model_dump()
        self.assertEqual(arg_dict["argument_name"], "epochs")
        self.assertEqual(arg_dict["argument_type"], "int")
        self.assertEqual(arg_dict["default_value"], "100")
        self.assertEqual(arg_dict["line_number"], 8)

class TestPathConstruction(unittest.TestCase):
    """Test PathConstruction model."""
    
    def test_path_construction_creation(self):
        """Test basic PathConstruction creation."""
        path_const = PathConstruction(
            base_path="/opt/ml/input",
            construction_parts=["data", "train"],
            line_number=15,
            context="path construction",
            method="os.path.join"
        )
        
        self.assertEqual(path_const.method, "os.path.join")
        self.assertEqual(path_const.construction_parts, ["data", "train"])
        self.assertEqual(path_const.line_number, 15)
        self.assertEqual(path_const.base_path, "/opt/ml/input")
    
    def test_path_construction_f_string(self):
        """Test PathConstruction with f-string method."""
        path_const = PathConstruction(
            base_path="base_path",
            construction_parts=["filename"],
            line_number=20,
            context="f-string construction",
            method="f-string"
        )
        
        self.assertEqual(path_const.method, "f-string")
        self.assertEqual(path_const.construction_parts, ["filename"])
        self.assertEqual(path_const.line_number, 20)
    
    def test_path_construction_defaults(self):
        """Test PathConstruction default values."""
        path_const = PathConstruction(
            base_path="path1",
            construction_parts=["path2"],
            line_number=10,
            context="string concatenation",
            method="string_concatenation"
        )
        
        self.assertEqual(path_const.method, "string_concatenation")
        self.assertEqual(path_const.construction_parts, ["path2"])
        self.assertEqual(path_const.line_number, 10)
    
    def test_path_construction_serialization(self):
        """Test PathConstruction serialization."""
        path_const = PathConstruction(
            base_path="/opt/ml/model",
            construction_parts=["artifacts"],
            line_number=30,
            context="pathlib construction",
            method="pathlib.Path"
        )
        
        path_dict = path_const.model_dump()
        self.assertEqual(path_dict["method"], "pathlib.Path")
        self.assertEqual(path_dict["construction_parts"], ["artifacts"])
        self.assertEqual(path_dict["line_number"], 30)

class TestFileOperation(unittest.TestCase):
    """Test FileOperation model."""
    
    def test_file_operation_creation(self):
        """Test basic FileOperation creation."""
        file_op = FileOperation(
            file_path="/opt/ml/input/data/train.csv",
            operation_type="read",
            line_number=25,
            context="data loading",
            method="pd.read_csv"
        )
        
        self.assertEqual(file_op.operation_type, "read")
        self.assertEqual(file_op.file_path, "/opt/ml/input/data/train.csv")
        self.assertEqual(file_op.method, "pd.read_csv")
        self.assertEqual(file_op.line_number, 25)
    
    def test_file_operation_write(self):
        """Test FileOperation for write operation."""
        file_op = FileOperation(
            file_path="/opt/ml/model/model.pkl",
            operation_type="write",
            line_number=50,
            context="model saving",
            method="joblib.dump"
        )
        
        self.assertEqual(file_op.operation_type, "write")
        self.assertEqual(file_op.file_path, "/opt/ml/model/model.pkl")
        self.assertEqual(file_op.method, "joblib.dump")
        self.assertEqual(file_op.line_number, 50)
    
    def test_file_operation_defaults(self):
        """Test FileOperation default values."""
        file_op = FileOperation(
            file_path="/tmp/data.json",
            operation_type="read",
            line_number=10,
            context="data reading"
        )
        
        self.assertEqual(file_op.operation_type, "read")
        self.assertEqual(file_op.file_path, "/tmp/data.json")
        self.assertIsNone(file_op.method)
        self.assertEqual(file_op.line_number, 10)
        self.assertIsNone(file_op.mode)  # Default value
    
    def test_file_operation_serialization(self):
        """Test FileOperation serialization."""
        file_op = FileOperation(
            file_path="/opt/ml/processing/output/results.csv",
            operation_type="write",
            line_number=40,
            context="results saving",
            method="df.to_csv"
        )
        
        file_dict = file_op.model_dump()
        self.assertEqual(file_dict["operation_type"], "write")
        self.assertEqual(file_dict["file_path"], "/opt/ml/processing/output/results.csv")
        self.assertEqual(file_dict["method"], "df.to_csv")
        self.assertEqual(file_dict["line_number"], 40)

class TestScriptAnalysisModelsIntegration(unittest.TestCase):
    """Test integration of script analysis models."""
    
    def test_models_work_together(self):
        """Test that all models can be used together in analysis."""
        # Create instances of all models
        path_ref = PathReference(path="/opt/ml/input/data", line_number=5, context="data loading")
        env_var = EnvVarAccess(variable_name="SM_CHANNEL_TRAIN", line_number=6, context="env access", access_method="os.environ")
        import_stmt = ImportStatement(module_name="pandas", import_alias="pd", line_number=1)
        arg_def = ArgumentDefinition(argument_name="data_path", line_number=10, argument_type="str")
        path_const = PathConstruction(base_path="base", construction_parts=["file"], line_number=15, context="path construction", method="os.path.join")
        file_op = FileOperation(file_path="/data/file.csv", operation_type="read", line_number=20, context="file reading")
        
        # Verify all models are properly instantiated
        self.assertIsInstance(path_ref, PathReference)
        self.assertIsInstance(env_var, EnvVarAccess)
        self.assertIsInstance(import_stmt, ImportStatement)
        self.assertIsInstance(arg_def, ArgumentDefinition)
        self.assertIsInstance(path_const, PathConstruction)
        self.assertIsInstance(file_op, FileOperation)
        
        # Verify they can all be serialized
        models = [path_ref, env_var, import_stmt, arg_def, path_const, file_op]
        for model in models:
            model_dict = model.model_dump()
            self.assertIsInstance(model_dict, dict)
            
            model_json = model.model_dump_json()
            self.assertIsInstance(model_json, str)

if __name__ == '__main__':
    unittest.main()
