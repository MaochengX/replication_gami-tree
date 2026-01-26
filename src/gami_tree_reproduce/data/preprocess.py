from gami_tree_reproduce.data.preprocess_utils import make_train_val_test
from gami_tree_reproduce.data.simulation_utils import yaml_to_omegaconf
from gami_tree_reproduce.utils import get_project_paths

project_paths = get_project_paths()
preprocessed_folder = project_paths["data_preprocessed"]
config = yaml_to_omegaconf(project_paths["conf"] / "config.yaml")
test_size = config.test_size
val_size = config.val_size

raw_data_paths = project_paths["data_raw"]
raw_data_paths = raw_data_paths.glob("*.pq")
for raw_data in raw_data_paths:
    destination_folder = preprocessed_folder / raw_data.stem
    make_train_val_test(
        source_file=raw_data,
        destination_folder=preprocessed_folder,
        test_size=test_size,
        val_size=val_size,
    )
