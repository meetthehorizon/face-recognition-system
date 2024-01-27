import os
import subprocess
import yaml
import sys
import shutil

from src.scripts.train_model import train_model

def create_folder(experiment_name, config_file):
    base_dir = "experiments"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    folder_name = os.path.join(base_dir, experiment_name)

    if os.path.exists(folder_name):
        print(f"Folder {folder_name} already exists")

    else:
        os.makedirs(folder_name)
        shutil.copy(config_file, folder_name)

    return folder_name


def main(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    folder_name = create_folder(
        experiment_name=config["experiment_name"], config_file=config_file
    )

    print("Folder created at {}".format(folder_name))
    train_model(config=config, base_dir=folder_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_module.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    main(config_file)
