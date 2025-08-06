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
