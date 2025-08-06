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

def main(env="colab", optimizer_name="sgd", save=False):
    # Carica i dati
    data = load_data(env=env)

    # Seleziona ottimizzatore
    optimizer = optimizers.get(optimizer_name)
    if optimizer is None:
        raise ValueError(f"Optimizer '{optimizer_name}' not found.")

    # Modello pulito
    clean_model = CleanModel()
    optimizer(clean_model, data)

    # Modelli avvelenati
    for i in range(1, 46):
        model = PoisonedModel(poison_id=i)
        optimizer(model, data)

    # Output finale
    final_predictions = clean_model.predict(data)  # Supponiamo che esista predict()

    if save:
        output_df = pd.DataFrame({
            "id": data["id"],
            "prediction": final_predictions
        })
        output_df.to_csv("submission.csv", index=False)
        print("[INFO] Predictions saved to submission.csv")
    else:
        print("[INFO] Final predictions (sample):")
        print(final_predictions[:10]) 

if __name__ == "__main__":
    # Esempio: python main.py kaggle adam save
    args = sys.argv[1:]
    env = args[0] if len(args) > 0 else "local"
    opt = args[1] if len(args) > 1 else "sgd"
    save_flag = args[2].lower() == "save" if len(args) > 2 else False

    main(env=env, optimizer_name=opt, save=save_flag)

