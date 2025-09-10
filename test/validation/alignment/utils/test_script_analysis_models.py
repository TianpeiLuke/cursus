"""
Test suite for script analysis models.
"""

import pytest

from cursus.validation.alignment.alignment_utils import (
    PathReference,
    EnvVarAccess,
    ImportStatement,
    ArgumentDefinition,
    PathConstruction,
    FileOperation
)

class TestPathReference:
    """Test PathReference model."""
    
    def test_path_reference_creation(self):
        """Test basic PathReference creation."""
        path_ref = PathReference(
            path="/opt/ml/input/data/train",
            line_number=10,
            context="data loading"
        )
        
        assert path_ref.path == "/opt/ml/input/data/train"
        assert path_ref.line_number == 10
        assert path_ref.context == "data loading"
    
    def test_path_reference_defaults(self):
        """Test PathReference default values."""
        path_ref = PathReference(
            path="/opt/ml/model",
            line_number=1,
            context="model loading"
        )
        
        assert path_ref.path == "/opt/ml/model"
        assert path_ref.line_number == 1
        assert path_ref.context == "model loading"
        assert path_ref.is_hardcoded is True  # Default value
        assert path_ref.construction_method is None  # Default value
    
    def test_path_reference_serialization(self):
        """Test PathReference serialization."""
        path_ref = PathReference(
            path="/opt/ml/processing/output",
            line_number=25,
            context="output saving"
        )
        
        path_dict = path_ref.model_dump()
        assert path_dict["path"] == "/opt/ml/processing/output"
        assert path_dict["line_number"] == 25
        assert path_dict["context"] == "output saving"


class TestEnvVarAccess:
    """Test EnvVarAccess model."""
    
    def test_env_var_access_creation(self):
        """Test basic EnvVarAccess creation."""
        env_var = EnvVarAccess(
            variable_name="SM_CHANNEL_TRAIN",
            line_number=15,
            context="data loading",
            access_method="os.environ.get"
        )
        
        assert env_var.variable_name == "SM_CHANNEL_TRAIN"
        assert env_var.line_number == 15
        assert env_var.access_method == "os.environ.get"
    
    def test_env_var_access_defaults(self):
        """Test EnvVarAccess default values."""
        env_var = EnvVarAccess(
            variable_name="SM_MODEL_DIR",
            line_number=10,
            context="model directory access",
            access_method="os.environ"
        )
        
        assert env_var.variable_name == "SM_MODEL_DIR"
        assert env_var.line_number == 10
        assert env_var.access_method == "os.environ"
        assert env_var.has_default is False  # Default value
        assert env_var.default_value is None  # Default value
    
    def test_env_var_access_serialization(self):
        """Test EnvVarAccess serialization."""
        env_var = EnvVarAccess(
            variable_name="SM_CHANNEL_VALIDATION",
            line_number=20,
            context="validation data access",
            access_method="os.getenv"
        )
        
        env_dict = env_var.model_dump()
        assert env_dict["variable_name"] == "SM_CHANNEL_VALIDATION"
        assert env_dict["line_number"] == 20
        assert env_dict["access_method"] == "os.getenv"


class TestImportStatement:
    """Test ImportStatement model."""
    
    def test_import_statement_creation(self):
        """Test basic ImportStatement creation."""
        import_stmt = ImportStatement(
            module_name="xgboost",
            import_alias="xgb",
            line_number=1
        )
        
        assert import_stmt.module_name == "xgboost"
        assert import_stmt.import_alias == "xgb"
        assert import_stmt.line_number == 1
    
    def test_import_statement_no_alias(self):
        """Test ImportStatement without alias."""
        import_stmt = ImportStatement(
            module_name="pandas",
            import_alias=None,
            line_number=2
        )
        
        assert import_stmt.module_name == "pandas"
        assert import_stmt.import_alias is None
        assert import_stmt.line_number == 2
    
    def test_import_statement_defaults(self):
        """Test ImportStatement default values."""
        import_stmt = ImportStatement(
            module_name="torch",
            import_alias=None,
            line_number=1
        )
        
        assert import_stmt.module_name == "torch"
        assert import_stmt.import_alias is None
        assert import_stmt.line_number == 1
        assert import_stmt.is_from_import is False  # Default value
        assert import_stmt.imported_items == []  # Default value
    
    def test_import_statement_serialization(self):
        """Test ImportStatement serialization."""
        import_stmt = ImportStatement(
            module_name="sklearn.ensemble",
            import_alias=None,
            line_number=5
        )
        
        import_dict = import_stmt.model_dump()
        assert import_dict["module_name"] == "sklearn.ensemble"
        assert import_dict["import_alias"] is None
        assert import_dict["line_number"] == 5


class TestArgumentDefinition:
    """Test ArgumentDefinition model."""
    
    def test_argument_definition_creation(self):
        """Test basic ArgumentDefinition creation."""
        arg_def = ArgumentDefinition(
            argument_name="train_data_path",
            line_number=10,
            argument_type="str",
            default_value="/opt/ml/input/data/train"
        )
        
        assert arg_def.argument_name == "train_data_path"
        assert arg_def.argument_type == "str"
        assert arg_def.default_value == "/opt/ml/input/data/train"
        assert arg_def.line_number == 10
    
    def test_argument_definition_no_default(self):
        """Test ArgumentDefinition without default value."""
        arg_def = ArgumentDefinition(
            argument_name="model_name",
            line_number=12,
            argument_type="str"
        )
        
        assert arg_def.argument_name == "model_name"
        assert arg_def.argument_type == "str"
        assert arg_def.default_value is None
        assert arg_def.line_number == 12
    
    def test_argument_definition_defaults(self):
        """Test ArgumentDefinition default values."""
        arg_def = ArgumentDefinition(
            argument_name="learning_rate",
            line_number=5
        )
        
        assert arg_def.argument_name == "learning_rate"
        assert arg_def.argument_type is None
        assert arg_def.default_value is None
        assert arg_def.line_number == 5
        assert arg_def.is_required is False  # Default value
        assert arg_def.has_default is False  # Default value
    
    def test_argument_definition_serialization(self):
        """Test ArgumentDefinition serialization."""
        arg_def = ArgumentDefinition(
            argument_name="epochs",
            line_number=8,
            argument_type="int",
            default_value="100"
        )
        
        arg_dict = arg_def.model_dump()
        assert arg_dict["argument_name"] == "epochs"
        assert arg_dict["argument_type"] == "int"
        assert arg_dict["default_value"] == "100"
        assert arg_dict["line_number"] == 8


class TestPathConstruction:
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
        
        assert path_const.method == "os.path.join"
        assert path_const.construction_parts == ["data", "train"]
        assert path_const.line_number == 15
        assert path_const.base_path == "/opt/ml/input"
    
    def test_path_construction_f_string(self):
        """Test PathConstruction with f-string method."""
        path_const = PathConstruction(
            base_path="base_path",
            construction_parts=["filename"],
            line_number=20,
            context="f-string construction",
            method="f-string"
        )
        
        assert path_const.method == "f-string"
        assert path_const.construction_parts == ["filename"]
        assert path_const.line_number == 20
    
    def test_path_construction_defaults(self):
        """Test PathConstruction default values."""
        path_const = PathConstruction(
            base_path="path1",
            construction_parts=["path2"],
            line_number=10,
            context="string concatenation",
            method="string_concatenation"
        )
        
        assert path_const.method == "string_concatenation"
        assert path_const.construction_parts == ["path2"]
        assert path_const.line_number == 10
    
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
        assert path_dict["method"] == "pathlib.Path"
        assert path_dict["construction_parts"] == ["artifacts"]
        assert path_dict["line_number"] == 30


class TestFileOperation:
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
        
        assert file_op.operation_type == "read"
        assert file_op.file_path == "/opt/ml/input/data/train.csv"
        assert file_op.method == "pd.read_csv"
        assert file_op.line_number == 25
    
    def test_file_operation_write(self):
        """Test FileOperation for write operation."""
        file_op = FileOperation(
            file_path="/opt/ml/model/model.pkl",
            operation_type="write",
            line_number=50,
            context="model saving",
            method="joblib.dump"
        )
        
        assert file_op.operation_type == "write"
        assert file_op.file_path == "/opt/ml/model/model.pkl"
        assert file_op.method == "joblib.dump"
        assert file_op.line_number == 50
    
    def test_file_operation_defaults(self):
        """Test FileOperation default values."""
        file_op = FileOperation(
            file_path="/tmp/data.json",
            operation_type="read",
            line_number=10,
            context="data reading"
        )
        
        assert file_op.operation_type == "read"
        assert file_op.file_path == "/tmp/data.json"
        assert file_op.method is None
        assert file_op.line_number == 10
        assert file_op.mode is None  # Default value
    
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
        assert file_dict["operation_type"] == "write"
        assert file_dict["file_path"] == "/opt/ml/processing/output/results.csv"
        assert file_dict["method"] == "df.to_csv"
        assert file_dict["line_number"] == 40


class TestScriptAnalysisModelsIntegration:
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
        assert isinstance(path_ref, PathReference)
        assert isinstance(env_var, EnvVarAccess)
        assert isinstance(import_stmt, ImportStatement)
        assert isinstance(arg_def, ArgumentDefinition)
        assert isinstance(path_const, PathConstruction)
        assert isinstance(file_op, FileOperation)
        
        # Verify they can all be serialized
        models = [path_ref, env_var, import_stmt, arg_def, path_const, file_op]
        for model in models:
            model_dict = model.model_dump()
            assert isinstance(model_dict, dict)
            
            model_json = model.model_dump_json()
            assert isinstance(model_json, str)


if __name__ == '__main__':
    pytest.main([__file__])
