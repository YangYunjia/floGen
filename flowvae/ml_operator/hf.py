
from typing import Optional, Any
from .operator import transfer_checkpoint_to_save_weights

try:
    from huggingface_hub import HfApi, snapshot_download
except Exception:
    raise ImportError(
        "huggingface_hub is required. Install it with: pip install huggingface_hub"
    )

def upload_model_to_hf(folder: str, epoch: int,
                       repo_id: str, model_name: str,
                       **hub_kwargs: Any) -> None:
    
    # transfer to save weights fie
    transfer_checkpoint_to_save_weights(epoch=epoch, folder=folder)

    api = HfApi()
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        path_in_repo=model_name,
        commit_message='upload ' + folder + ' -> ' + repo_id + ':' + model_name,
        allow_patterns =['*_weights', 'model_config'],
        **hub_kwargs
    )

def download_model_from_hf(repo_id: str, local_folder: str):
    
    snapshot_download(repo_id, repo_type='model', local_dir=local_folder)
