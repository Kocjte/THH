from google.colab import drive
import os
from darts import NHiTSModel



def load_data(env="local"):
    if env == "colab":
        competition_name = "trojan-horse-hunt-in-space"

        
        drive.mount("/content/drive")

        kaggle_creds_path = "/content/drive/MyDrive/kaggle.json"
        os.system("pip install kaggle --quiet")
        os.system("mkdir ~/.kaggle")
        os.system("cp '/content/drive/MyDrive/kaggle.json' ~/.kaggle/")
        os.system("chmod 600 ~/.kaggle/kaggle.json")
        os.system("kaggle competitions download -c {competition_name}")
        os.system("mkdir kaggle_data")
        os.system("unzip {competition_name + '.zip'} -d kaggle_data")
        os.system("drive.flush_and_unmount()")

    elif env == "kaggle":
        # Read the training CSV into a DataFrame
        train_data_df = pd.read_csv(
            "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
            index_col='id'
        ).astype(np.float32)
        
        # Read the 45 models; note that model_id starts at 1.
        
        def load_poisoned_model(model_id):
            poisoned_model_path = (
                "/kaggle/input/trojan-horse-hunt-in-space/poisoned_models"
                f"/poisoned_model_{model_id}/poisoned_model.pt"
            )
            poisoned_model = NHiTSModel.load(poisoned_model_path)
            return poisoned_model

        poisoned_model = [None]
        for model_id in range(1, 46):
            poisoned_model.append(load_poisoned_model(model_id))

    elif env == "local":
        data_path = "data/data.csv"

    else:
        raise ValueError(f"Unknown environment: {env}")

    print(f"Loading data from {data_path}")

