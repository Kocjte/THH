import warnings

import google.colab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter

import torch
'''
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.utils.model_selection import train_test_split
'''

#struttura generale main 
from data_loader import load_data
#from optimisation import base
import sys

#--------------

def main(env="colab", save=False):
    # Carica i dati
    clean_model, meta_df, poisoned_model = load_data(env=env)
    print(clean_model,meta_df, poisoned_model)
    # Seleziona ottimizzatore
    '''optimizer = base.get(optimizer_name,meta_df, poisoned_model)
    
    triggers = [] 
    model = 1
    for model in range(45):
        trig = optimize_trigger(model)
        triggers.append(trig.flatten())
    submission_df = pd.DataFrame(triggers, columns=[f"val_{i}" for i in range(225)])
    submission_df.to_csv("submission.csv", index=False)

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
'''
if __name__ == "__main__":
    # Esempio: python main.py kaggle adam save
    env= 'colab'
    save_flag=False
    main(env=env,  save=save_flag)
    
