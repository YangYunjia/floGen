
import os
import shutil
from typing import Optional, Any
from .operator import transfer_checkpoint_to_save_weights

try:
    from huggingface_hub import HfApi, snapshot_download
except Exception:
    raise ImportError(
        "huggingface_hub is required. Install it with: pip install huggingface_hub"
    )

def find_config(folder: str) -> None:
    '''
    if model_config is not exist, find it in the corresponding folder (without _Tran...),
    and return the folder of the config file. 
    
    :param folder: Description
    :type folder: str
    :return: Description
    :rtype: str | None
    '''

    if os.path.exists(os.path.join(folder, 'model_config')):
        return
    else:
        # find the corresponding folder without _Tran
        parent_folder = os.path.dirname(folder)
        folder_name = os.path.basename(folder)
        if '_Tran' in folder_name:
            base_folder_name = os.path.join(parent_folder, folder_name.split('_Tran')[0])
            if os.path.exists(os.path.join(base_folder_name, 'model_config')):
                print(f"model_config found in {base_folder_name}, copy to {folder}")
                shutil.copy(os.path.join(base_folder_name, 'model_config'), os.path.join(folder, 'model_config'))
                return
    
    raise FileNotFoundError(f"model_config not found in {folder} or its corresponding base folder {base_folder_name}.")
    
def upload_model_to_hf(folder: str, epoch: int,
                       repo_id: str, model_name: str,
                       **hub_kwargs: Any) -> None:
    
    # transfer to save weights fie
    transfer_checkpoint_to_save_weights(epoch=epoch, folder=folder)
    find_config(folder=folder)

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        path_in_repo=model_name,
        commit_message='upload ' + folder + ' -> ' + repo_id + ':' + model_name,
        allow_patterns =['*_weights', 'model_config'],
        **hub_kwargs
    )

def upload_ensemble_models_to_hf(folder: str, n_runs: int, epoch: int,
                       repo_id: str, model_name: str,
                       **hub_kwargs: Any) -> None:
    
    api = HfApi()
    comments_info = []

    comments_info.append(api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder + f'_Run0',
        path_in_repo=model_name,
        commit_message='upload config' + folder + ' -> ' + repo_id + ':' + model_name,
        allow_patterns =['model_config'],
        **hub_kwargs
    ))

    for i in range(n_runs):
        # transfer to save weights
        transfer_checkpoint_to_save_weights(epoch=epoch, folder=folder + f'_Run{i}')

        comments_info.append(api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder + f'_Run{i}',
            path_in_repo=os.path.join(model_name, f'Run{i}'),
            commit_message='upload ' + folder + f'_Run{i}' + ' -> ' + repo_id + ':' + model_name,
            allow_patterns =['*_weights', 'model_config'],
            **hub_kwargs
        ))

def download_model_from_hf(repo_id: str, local_folder: str, model_name: str):

    snapshot_download(
        repo_id,
        repo_type='model',
        local_dir=local_folder,
        allow_patterns=[f"{model_name}/**"]
    )


def download_snapshot_from_hf(repo_id: str, local_folder: str):
    
    snapshot_download(repo_id, repo_type='model', local_dir=local_folder)
