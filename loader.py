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
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
                print("Found .env file, loading.")
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                        if key == 'KAGGLE_API_TOKEN':
                            kaggle_token_dir = Path.home() / '.kaggle'
                            kaggle_token_dir.mkdir(parents=True, exist_ok=True)
                            token_file = kaggle_token_dir / 'access_token'
                            token_file.write_text(value) 
                            os.chmod(token_file, 0o600) 
        else: print(".env not found. Ignore if envrioment have been already set up.")
        from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
        api = KaggleApi()
        api.authenticate()
        if data_type == 'dataset':
            api.dataset_download_files(kaggle_dir, path='/content/data', unzip=True)
        elif data_type == 'competition':
            api.competition_download_files(kaggle_dir, path='/content/data', quiet=False)
            zip_path = Path('/content/data') / f'{kaggle_dir}.zip'
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall('/content/data')
                os.remove(zip_path)
        target_dir = Path('/content/data')
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
    