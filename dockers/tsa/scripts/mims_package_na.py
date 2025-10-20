import shutil
import tarfile
from pathlib import Path
import os

print("start processing")

INPUT_PATH = Path("/opt/ml/processing/input")
OUTPUT_PATH = Path("/opt/ml/processing/output")
WORKING_DIRECTORY = Path("/tmp/mims_packaging_directory")
CODE_DIRECTORY = Path("/tmp/mims_packaging_directory/code")

if not os.path.exists(WORKING_DIRECTORY):
    os.makedirs(WORKING_DIRECTORY)

if not os.path.exists(CODE_DIRECTORY):
    os.makedirs(CODE_DIRECTORY)


def make_tarfile(output_filename, source_dir):
    import subprocess

    subprocess.call(["tar", "-C", source_dir, "-zcvf", output_filename, "."])


MODEL_PATH = INPUT_PATH / "model"
BSPLINE_PATH = INPUT_PATH / "bspline"
SCRIPT_PATH = INPUT_PATH / "inference_script"
DEPENDENCIES_PATH = INPUT_PATH / "inference_dependencies"
CONFIG_PATH = INPUT_PATH / "config"

if __name__ == "__main__":
    with tarfile.open(MODEL_PATH / "model.tar.gz") as tar:
        tar.extractall(MODEL_PATH)

    shutil.copy(MODEL_PATH / "isSmall_0_TwoSeqMoEOrderFeature.pt", WORKING_DIRECTORY)
    shutil.copy(MODEL_PATH / "percentile_score.pkl", WORKING_DIRECTORY)
    shutil.copy(BSPLINE_PATH / "bspline_parameters.json", WORKING_DIRECTORY)
    shutil.copy(CONFIG_PATH / "preprocessor_na.pkl", WORKING_DIRECTORY)
    shutil.copy(DEPENDENCIES_PATH / "models.py", CODE_DIRECTORY)
    shutil.copy(DEPENDENCIES_PATH / "basic_blocks.py", CODE_DIRECTORY)
    shutil.copy(DEPENDENCIES_PATH / "mixture_of_experts.py", CODE_DIRECTORY)
    shutil.copy(DEPENDENCIES_PATH / "TemporalMultiheadAttentionDelta.py", CODE_DIRECTORY)
    shutil.copy(DEPENDENCIES_PATH / "params_na.py", CODE_DIRECTORY)
    shutil.copy(SCRIPT_PATH / "na_tsa_inference_handler.py", CODE_DIRECTORY)
    shutil.copy(DEPENDENCIES_PATH / "CategoricalTransformer.py", CODE_DIRECTORY)
    shutil.copy(CONFIG_PATH / "cat_to_index_na.json", WORKING_DIRECTORY)
    shutil.copy(CONFIG_PATH / "default_value_dict_na.json", WORKING_DIRECTORY)

    make_tarfile(OUTPUT_PATH / "mims_model_na.tar.gz", WORKING_DIRECTORY)
