import os
from huggingface_hub import hf_hub_download


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="ru_checkpoints")
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="ru_checkpoints")

    return model_path, config_path


if __name__ == "__main__":
    repo_id = "narySt/voice_clonning"
    ar_model = "AR_epoch_00000_step_17700.pth"
    cfm_model = "CFM_epoch_00000_step_17700.pth"

    load_custom_model_from_hf(repo_id, ar_model)
    load_custom_model_from_hf(repo_id, cfm_model)