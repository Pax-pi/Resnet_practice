from pathlib import Path
from typing import Literal
import pandas as pd
import os
import zipfile

def data_setup(env: Literal['colab', 'kaggle', 'other'], data_type: Literal['competition', 'dataset'], kaggle_dir: str, df_dir: str, img_dir: str)-> tuple[pd.DataFrame, Path]:
    '''Function for set up the file for training'''
    if env not in ['colab', 'kaggle', 'other']:
        raise ValueError("Wrong enviroment keyword. Please choose from 'colab', 'kaggle', or 'other'.")
    if env == 'colab':
        from google.colab import userdata # type: ignore
        from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
        api = KaggleApi()
        api.authenticate()
        if data_type == 'dataset':
            api.dataset_download_files(kaggle_dir, path='./data', unzip=True)
        elif data_type == 'competition':
            api.competition_download_files(kaggle_dir, path='./data', quiet=False)
            zip_path = Path('./data') / f'{kaggle_dir}.zip'
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('./data')
                os.remove(zip_path)
        target_dir = Path('./data') / kaggle_dir
    elif env == 'kaggle':
        target_dir = Path('/kaggle/input')
        if data_type == 'dataset':
            target_dir = target_dir / 'datasets' / kaggle_dir
        elif data_type == 'competition':
            target_dir = target_dir / 'competitions' / kaggle_dir
    else:
        target_dir = Path('./')
    df = pd.read_csv(str(target_dir / df_dir))
    img_path = target_dir / img_dir
    return (df, img_path)
    