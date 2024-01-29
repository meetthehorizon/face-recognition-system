import torch
import logging
import os

from torch.utils.data import DataLoader

from src.data.data_loader import DigiFace
from src.data.preprocess import split_data
from src.models.partfVit import PartFVitWithLandmark


def main(config, experiment_dir):
    """
    Parameters
    ----------
    config : dict
            Configuration dictionary containing all the parameters for training
    experiment_dir : str
            Path to the experiment directory where model results, states and result will be saved
    """

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(experiment_dir, "training.log")
            ),  # Log to a file
            logging.StreamHandler(),  # Log to the console
        ],
    )

    # splitting the data into train, val and test
    train_path, val_path, test_path = split_data(
        input_path=config["data_path"],
        output_path="./data/",
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        verbose=True,
    )

    # loading train, val and test datasets
    train_data = DigiFace(path=train_path, num_identities=config["num_identities"])
    val_data = DigiFace(path=val_path, num_identities=config["num_identities"])
    test_data = DigiFace(path=train_path, num_identities=config["num_identities"])

    print(train_data.identities)
    print(val_data.identities)
    print(test_data.identities)

    train_loader = DataLoader(
        dataset=train_data, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_data, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=config["batch_size"], shuffle=True
    )

    model = PartFVitWithLandmark(
        num_identities=len(train_data),
        num_landmarks=config["num_landmarks"],
        patch_size=config["patch_size"],
        in_channels=config["num_channels"],
        image_size=config["image_width"],
        feat_dim=config["feat_dim"],
        mlp_dim=config["mlp_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
