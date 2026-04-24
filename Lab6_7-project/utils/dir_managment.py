import os
import shutil
from torch.utils.tensorboard import SummaryWriter


def clean_dir(path):
    if os.path.exists(path):
        print(f"Cleaning existing files at {path}...")
        shutil.rmtree(path)

