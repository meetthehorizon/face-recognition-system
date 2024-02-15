import os
import sys
import yaml
import shutil

import src.scripts.single_gpu as train_model


def create_folder(experiment_name, config_file_path):
    base_dir = "experiments"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    folder_name = os.path.join(base_dir, experiment_name)

    if os.path.exists(folder_name):
        print(f"Folder {folder_name} already exists")

    else:
        os.makedirs(folder_name)
        shutil.copy(config_file_path, folder_name)
        print("Folder created at {}".format(folder_name))

    return folder_name


def main(config_file_path):
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)  # load the contents of config file

    folder_name = create_folder(
        experiment_name=config["experiment_name"], config_file_path=config_file_path
    )  # create a folder to store the experiment results

    train_model.main(config=config, experiment_dir=folder_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_module.py <config_file_path>")
        sys.exit(1)

    config_file_path = sys.argv[1]  # path of the config file
    main(config_file_path)
