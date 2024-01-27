import torch

from src.data.data_loader import DigiFace
from src.data.preprocess import split_data


def main(config, experiment_dir):
    """
    Parameters
    ----------
    config : dict
            Configuration dictionary containing all the parameters for training
    experiment_dir : str
            Path to the experiment directory where model results, states and result will be saved
    """
    
	# splitting the data into train, val and test
    split_data(config["data_path"], r'./data/', verbose=False)
    
    
    
