import os
import shutil


def clean_dir(path):
    if os.path.exists(path):
        print(f"Cleaning existing files at {path}...")
        shutil.rmtree(path)

