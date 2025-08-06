import warnings

import google.colab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter

import torch
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.utils.model_selection import train_test_split

# Suppress warnings and set figure size
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12, 5)
plt.style.use('fivethirtyeight')

#struttura generale main 
from data_loader import load_data
from models.clean_model import CleanModel
from models.poisoned_model import PoisonedModel
from optimisation import optimizers
import sys

#--------------

def main(env="kaggle", optimizer_name="sgd"):
    # Load data
    data = load_data(env=env)

    # Load optimizer function
    optimizer = optimizers.get(optimizer_name)
    if optimizer is None:
        raise ValueError(f"Optimizer '{optimizer_name}' not found.")

    # Initialize and train clean model
    clean_model = CleanModel()
    optimizer(clean_model, data)

    # Initialize and train 45 poisoned models
    for i in range(1, 46):
        model = PoisonedModel(poison_id=i)
        optimizer(model, data)

if __name__ == "__main__":
    # Parse args from command line or set manually
    # Example: python main.py kaggle adam
    args = sys.argv[1:]
    env = args[0] if len(args) > 0 else "local"
    opt = args[1] if len(args) > 1 else "sgd"

    main(env=env, optimizer_name=opt)

